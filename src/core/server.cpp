#include <arpa/inet.h>
#include <array>
#include <condition_variable>
#include <csignal>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <optional>
#include <queue>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "core/cnn.hpp"
#include "data/constants.hpp"
#include "data/mnist_dataset.hpp"

using json = nlohmann::json;

#define MAX_EVENTS 64
#define DEFAULT_PORT 8080
#define THREAD_POOL_SIZE 4
#define HEADER_BUFFER_SIZE 4096

static volatile sig_atomic_t keep_running = 1;

static void handle_signal(int)
{
    keep_running = 0;
}

class ThreadPool
{
public:
    explicit ThreadPool(size_t threads) : stop(false)
    {
        for (size_t i = 0; i < threads; ++i)
        {
            workers.emplace_back([this] {
                for (;;)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F &&f)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

static void set_non_blocking(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static std::optional<std::string> read_file(const std::string &path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        return std::nullopt;

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string buffer(size, '\0');
    if (file.read(buffer.data(), size))
    {
        return buffer;
    }
    return std::nullopt;
}

static void send_response(int client_fd, const std::string &status,
                          const std::string &content_type,
                          const std::string &body)
{
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.length() << "\r\n"
        << "Connection: close\r\n"
        << "Access-Control-Allow-Origin: *\r\n\r\n"
        << body;
    std::string response = oss.str();
    send(client_fd, response.data(), response.length(), 0);
}

static bool parse_pixels(const std::string &body, Tensor &image, const CNN &cnn)
{
    try
    {
        json j = json::parse(body);
        if (!j.contains("pixels") || !j["pixels"].is_array())
            return false;

        const auto &pixels = j["pixels"];
        if (pixels.size() != PIXELS)
            return false;

        for (size_t i = 0; i < PIXELS; i++)
        {
            scalar_t raw = static_cast<scalar_t>(pixels[i]) / NORMALIZE_DIVISOR;
            size_t row = i / IMG_HEIGHT;
            size_t col = i % IMG_HEIGHT;
            image(row * IMG_HEIGHT + col, 0) = cnn.normalize()
                ? (raw - cnn.mean()) / cnn.std()
                : raw;
        }
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static void handle_predict(int client_fd, const CNN &cnn, const std::string &body)
{
    Tensor image(Shape({ PIXELS, 1 }), 0.0f);
    if (!parse_pixels(body, image, cnn))
    {
        send_response(client_fd, "400 Bad Request", "text/plain",
                      "malformed pixels\n");
        return;
    }

    std::vector<scalar_t> probabilities = cnn.predict(image);

    int prediction = 0;
    scalar_t max_prob = probabilities[0];
    for (size_t i = 1; i < probabilities.size(); i++)
    {
        if (probabilities[i] > max_prob)
        {
            max_prob = probabilities[i];
            prediction = static_cast<int>(i);
        }
    }

    std::ostringstream response;
    response << "{\"prediction\":" << prediction << ",\"probabilities\":[";
    for (size_t i = 0; i < probabilities.size(); i++)
    {
        response << (i == 0 ? "" : ",") << std::fixed
                 << std::setprecision(10) << probabilities[i];
    }
    response << "]}\n";

    send_response(client_fd, "200 OK", "application/json", response.str());
}

static void handle_client(int client_fd, const CNN &cnn)
{
    std::string request;
    char buffer[HEADER_BUFFER_SIZE];
    ssize_t received = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (received <= 0)
    {
        close(client_fd);
        return;
    }
    buffer[received] = '\0';
    request.append(buffer, received);

    size_t body_pos = request.find("\r\n\r\n");
    if (body_pos == std::string::npos)
    {
        close(client_fd);
        return;
    }

    std::string headers = request.substr(0, body_pos);
    std::string body = request.substr(body_pos + 4);

    std::istringstream header_stream(headers);
    std::string method, path, version;
    header_stream >> method >> path >> version;

    if (method == "GET")
    {
        auto page = read_file("public/index.html");
        if (page)
            send_response(client_fd, "200 OK", "text/html", *page);
        else
            send_response(client_fd, "404 Not Found", "text/plain", "not found");
    }
    else if (method == "POST" && path == "/predict")
    {
        size_t content_length = 0;
        std::string line;
        while (std::getline(header_stream, line) && line != "\r")
        {
            if (line.find("Content-Length:") == 0)
            {
                content_length = std::stoul(line.substr(15));
            }
        }

        while (body.length() < content_length)
        {
            received = recv(client_fd, buffer, sizeof(buffer), 0);
            if (received <= 0)
                break;
            body.append(buffer, received);
        }
        handle_predict(client_fd, cnn, body);
    }
    close(client_fd);
}

int main(int argc, char **argv)
{
    std::string model_path = argc > 1 ? argv[1] : "mnist_cnn.model";
    int port = argc > 2 ? std::stoi(argv[2]) : DEFAULT_PORT;
    CNN cnn;
    try
    {
        cnn.load_from_model(model_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return 1;
    }

    signal(SIGINT, handle_signal);

    omp_set_num_threads(1);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int enabled = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
    set_non_blocking(server_fd);
    listen(server_fd, SOMAXCONN);

    int epoll_fd = epoll_create1(0);
    struct epoll_event ev, events[MAX_EVENTS];
    ev.events = EPOLLIN | EPOLLET;
    ev.data.fd = server_fd;
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &ev);

    ThreadPool pool(THREAD_POOL_SIZE);

    std::cout << "MNIST server running on port " << port
             << " (epoll + thread pool)\n";

    while (keep_running)
    {
        int n_fds = epoll_wait(epoll_fd, events, MAX_EVENTS, 1000);
        if (n_fds <= 0)
            continue;

        for (int i = 0; i < n_fds; ++i)
        {
            if (events[i].data.fd == server_fd)
            {
                while (true)
                {
                    struct sockaddr_in client_addr;
                    socklen_t client_len = sizeof(client_addr);
                    int client_fd = accept(
                        server_fd, (struct sockaddr *)&client_addr, &client_len);
                    if (client_fd == -1)
                        break;

                    set_non_blocking(client_fd);
                    ev.events = EPOLLIN | EPOLLONESHOT;
                    ev.data.fd = client_fd;
                    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);
                }
            }
            else
            {
                int client_fd = events[i].data.fd;
                pool.enqueue([client_fd, &cnn]() {
                    handle_client(client_fd, cnn);
                });
            }
        }
    }

    close(server_fd);
    close(epoll_fd);
    return 0;
}
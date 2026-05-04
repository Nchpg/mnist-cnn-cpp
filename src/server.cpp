#include "cnn.hpp"
#include "mnist_dataset.hpp"
#include "constants.hpp"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <signal.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <optional>
#include <array>
#include <iomanip>

#define HEADER_BUFFER_SIZE 4096
#define DEFAULT_PORT 8080

static volatile sig_atomic_t keep_running = 1;

static void handle_signal(int) {
    keep_running = 0;
}

static std::optional<std::string> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return std::nullopt;
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::string buffer(size, '\0');
    if (file.read(buffer.data(), size)) {
        return buffer;
    }
    return std::nullopt;
}

static void send_response(int client_fd, const std::string& status, const std::string& content_type, const std::string& body) {
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

static bool parse_pixels(const std::string& body, Matrix& image) {
    size_t start = body.find('[');
    if (start == std::string::npos) return false;
    
    std::string arr = body.substr(start + 1);
    std::vector<scalar_t> web_pixels;
    web_pixels.reserve(28 * 28);
    
    std::stringstream ss(arr);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        try {
            scalar_t pixel = static_cast<scalar_t>(std::stod(token));
            web_pixels.push_back(pixel);
            if (web_pixels.size() == PIXELS) break;
        } catch (...) {
            return false;
        }
    }

    if (web_pixels.size() != PIXELS) return false;

    for (size_t row = 0; row < IMG_HEIGHT; row++) {
        for (size_t col = 0; col < IMG_WIDTH; col++) {
            scalar_t raw = web_pixels[col * IMG_HEIGHT + row] / NORMALIZE_DIVISOR;
            image(row * IMG_HEIGHT + col, 0) = MnistDataset::normalize() ? (raw - MnistDataset::mean()) / MnistDataset::std() : raw;
        }
    }
    return true;
}

static void handle_predict(int client_fd, CNN& cnn, const std::string& body) {
    Matrix image(PIXELS, 1);
    if (!parse_pixels(body, image)) {
        send_response(client_fd, "400 Bad Request", "text/plain", "malformed pixels\n");
        return;
    }

    std::vector<scalar_t> probabilities = cnn.predict(image);
    int prediction = cnn.predict_label(image);

    std::ostringstream response;
    response << "{\"prediction\":" << prediction << ",\"probabilities\":[";
    for (size_t i = 0; i < probabilities.size(); i++) {
        response << (i == 0 ? "" : ",") << std::fixed << std::setprecision(10) << probabilities[i];
    }
    response << "]}\n";

    send_response(client_fd, "200 OK", "application/json", response.str());
}

static void handle_client(int client_fd, CNN& cnn) {
    std::string request;
    char buffer[HEADER_BUFFER_SIZE];
    ssize_t received = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (received <= 0) return;
    buffer[received] = '\0';
    request.append(buffer, received);

    size_t body_pos = request.find("\r\n\r\n");
    if (body_pos == std::string::npos) return;

    std::string headers = request.substr(0, body_pos);
    std::string body = request.substr(body_pos + 4);

    std::istringstream header_stream(headers);
    std::string method, path, version;
    header_stream >> method >> path >> version;

    if (method == "GET") {
        auto page = read_file("public/index.html");
        if (page) send_response(client_fd, "200 OK", "text/html", *page);
        else send_response(client_fd, "404 Not Found", "text/plain", "not found");
        return;
    }

    if (method == "POST" && path == "/predict") {
        size_t content_length = 0;
        std::string line;
        while (std::getline(header_stream, line) && line != "\r") {
            if (line.find("Content-Length:") == 0) {
                content_length = std::stoul(line.substr(15));
            }
        }

        while (body.length() < content_length) {
            received = recv(client_fd, buffer, sizeof(buffer), 0);
            if (received <= 0) break;
            body.append(buffer, received);
        }
        handle_predict(client_fd, cnn, body);
    }
}

static int create_server_socket(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int enabled = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(server_fd, 16);
    return server_fd;
}

int main(int argc, char **argv) {
    std::string model_path = argc > 1 ? argv[1] : "mnist_cnn.model";
    int port = argc > 2 ? std::stoi(argv[2]) : DEFAULT_PORT;
    CNN cnn;
    try {
        cnn.load_from_model(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return 1;
    }
    signal(SIGINT, handle_signal);
    int server_fd = create_server_socket(port);
    std::cout << "MNIST server on port " << port << std::endl;
    while (keep_running) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd >= 0) {
            handle_client(client_fd, cnn);
            close(client_fd);
        }
    }
    close(server_fd);
    return 0;
}

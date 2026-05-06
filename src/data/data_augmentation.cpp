#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "data/constants.hpp"

using namespace std;
using namespace cv;

bool parseCSVLine(const string& line, uint8_t& label, Mat& image)
{
    stringstream ss(line);
    string val;

    if (!getline(ss, val, ','))
        return false;
    label = static_cast<uint8_t>(stoi(val));

    image = Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC1);
    int i = 0;
    while (getline(ss, val, ','))
    {
        if (i < static_cast<int>(PIXELS))
        {
            image.at<uint8_t>(i / IMG_WIDTH, i % IMG_WIDTH) = static_cast<uint8_t>(stoi(val));
            i++;
        }
    }
    return i == static_cast<int>(PIXELS);
}

Mat augmentImage(const Mat& input, mt19937& gen)
{
    Mat output;

    uniform_real_distribution<> angleDist(-15.0, 15.0);
    uniform_real_distribution<> shiftDist(-3.0, 3.0);
    uniform_real_distribution<> zoomDist(0.85, 1.15);

    double angle = angleDist(gen);
    double tx = shiftDist(gen);
    double ty = shiftDist(gen);
    double scale = zoomDist(gen);

    Point2f center((input.cols - 1) / 2.0, (input.rows - 1) / 2.0);

    Mat rot = getRotationMatrix2D(center, angle, scale);

    rot.at<double>(0, 2) += tx;
    rot.at<double>(1, 2) += ty;

    warpAffine(input, output, rot, input.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

    return output;
}

void writeBinaryRecord(ofstream& outFile, uint8_t label, const Mat& image)
{
    outFile.write(reinterpret_cast<const char*>(&label), sizeof(label));

    if (image.isContinuous())
    {
        outFile.write(reinterpret_cast<const char*>(image.data), 784);
    }
    else
    {
        Mat continuousImg = image.clone();
        outFile.write(reinterpret_cast<const char*>(continuousImg.data), 784);
    }
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        cerr << "Usage: " << argv[0] << " <source.csv> <destination.bin> <nb_copies>" << endl;
        cerr << "Example: " << argv[0] << " emnist-mnist-train.csv emnist_augmented.bin 9" << endl;
        return -1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];
    int augmentationsPerImage = 0;

    try
    {
        augmentationsPerImage = stoi(argv[3]);
    }
    catch (...)
    {
        cerr << "Erreur: Le nombre de copies doit être un entier valide." << endl;
        return -1;
    }

    ifstream inFile(inputFile);
    ofstream outFile(outputFile, ios::binary);

    if (!inFile.is_open())
    {
        cerr << "Erreur: Impossible d'ouvrir le fichier source (" << inputFile << ")" << endl;
        return -1;
    }
    if (!outFile.is_open())
    {
        cerr << "Erreur: Impossible de créer le fichier de destination (" << outputFile << ")" << endl;
        return -1;
    }

    string line;
    uint8_t label;
    Mat image;

    random_device rd;
    mt19937 gen(rd());

    int count = 0;
    cout << "Traitement en cours..." << endl;

    streampos oldPos = inFile.tellg();
    if (getline(inFile, line))
    {
        if (line.find(',') != string::npos && !isdigit(line[0]))
        {
        }
        else
        {
            inFile.seekg(oldPos);
        }
    }

    while (getline(inFile, line))
    {
        if (line.empty())
            continue;

        if (parseCSVLine(line, label, image))
        {
            writeBinaryRecord(outFile, label, image);

            for (int i = 0; i < augmentationsPerImage; ++i)
            {
                Mat augImage = augmentImage(image, gen);
                writeBinaryRecord(outFile, label, augImage);
            }

            count++;
            if (count % 1000 == 0)
            {
                cout << count << " images originales traitées (" << count * (augmentationsPerImage + 1)
                     << " images générées)..." << flush << "\r";
            }
        }
    }

    cout << "\nTerminé ! " << count << " images originales traitées au total." << endl;
    cout << "Fichier généré : " << outputFile << " (" << count * (augmentationsPerImage + 1) << " images)" << endl;

    inFile.close();
    outFile.close();

    return 0;
}
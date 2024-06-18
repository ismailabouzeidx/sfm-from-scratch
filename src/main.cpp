#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace cv::viz;
using namespace std;

int main() {
    // Intrinsic matrix
    Matx33d K(1520.40, 0, 302.32,
              0, 1525.90, 246.87,
              0, 0, 1);

    // Load images
    vector<string> imagePaths;
    for (int i = 1; i <= 10; ++i) {
        imagePaths.push_back("/home/ismail/temple/" + to_string(i) + ".png");
    }

    // SIFT detector and descriptor
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Viz window
    viz::Viz3d window("Viz Window");
    window.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    Mat prevImg, currImg;
    vector<KeyPoint> prevKeypoints, currKeypoints;
    Mat prevDescriptors, currDescriptors;
    Mat R, t;
    Affine3d pose = Affine3d::Identity();

    for (size_t i = 0; i < imagePaths.size(); ++i) {
        // Load current image
        currImg = imread(imagePaths[i], IMREAD_GRAYSCALE);

        if (currImg.empty()) {
            cerr << "Could not load image at: " << imagePaths[i] << endl;
            return -1;
        }

        // Detect and compute SIFT features
        sift->detectAndCompute(currImg, noArray(), currKeypoints, currDescriptors);

        if (i > 0) { // Skip the first image
            // Match descriptors using FLANN-based matcher
            FlannBasedMatcher matcher;
            vector<vector<DMatch>> knnMatches;
            matcher.knnMatch(prevDescriptors, currDescriptors, knnMatches, 2);

            // Filter good matches using Lowe's ratio test
            vector<DMatch> good_matches;
            for (size_t j = 0; j < knnMatches.size(); j++) {
                if (knnMatches[j][0].distance < 0.7 * knnMatches[j][1].distance) {
                    good_matches.push_back(knnMatches[j][0]);
                }
            }

            // Extract matched keypoints
            vector<Point2f> prevPts, currPts;
            for (size_t j = 0; j < good_matches.size(); j++) {
                prevPts.push_back(prevKeypoints[good_matches[j].queryIdx].pt);
                currPts.push_back(currKeypoints[good_matches[j].trainIdx].pt);
            }

            // Find essential matrix
            Mat E = findEssentialMat(currPts, prevPts, K, RANSAC);

            // Recover pose
            recoverPose(E, currPts, prevPts, K, R, t);

            // Update pose
            pose = pose * Affine3d(R, t);

            // Display current pose
            window.showWidget("Camera" + to_string(i), viz::WCameraPosition(K, currImg, 1.0, viz::Color::blue()));
            window.setWidgetPose("Camera" + to_string(i), pose);
        }

        // Show the current image
        cv::imshow("Image", currImg);
        waitKey(100);

        // Move to next image
        prevImg = currImg.clone();
        prevKeypoints = currKeypoints;
        prevDescriptors = currDescriptors;
    }

    // Display the viz window
    window.spin();

    return 0;
}

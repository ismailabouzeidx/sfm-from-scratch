#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <iostream>
#include <X11/Xlib.h>  // Include the Xlib header

int main() {
    // Initialize Xlib in a thread-safe manner
    XInitThreads();

    // Intrinsics matrix
    cv::Matx33d K( 1520.40, 0, 302.32,
                  0, 1525.90, 246.87,
                  0, 0, 1);

    // Read grayscale images for feature detection
    cv::Mat img1_gray = cv::imread("/home/ismail/1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2_gray = cv::imread("/home/ismail/2.png", cv::IMREAD_GRAYSCALE);

    // Read color images for visualization
    cv::Mat img1_color = cv::imread("/home/ismail/1.png", cv::IMREAD_COLOR);
    cv::Mat img2_color = cv::imread("/home/ismail/2.png", cv::IMREAD_COLOR);

    if (img1_gray.empty() || img2_gray.empty() ||
        img1_color.empty() || img2_color.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return -1;
    }

    // Detect SIFT features and compute descriptors
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(img1_gray, cv::Mat(), keypoints1, descriptors1);
    sift->detectAndCompute(img2_gray, cv::Mat(), keypoints2, descriptors2);

    // Match descriptors between img2 and img1 using FLANN based matcher
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches2_1;
    matcher.knnMatch(descriptors2, descriptors1, knn_matches2_1, 2);

    // Perform Lowe's ratio test to filter matches
    std::vector<cv::DMatch> good_matches2_1;
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knn_matches2_1.size(); ++i) {
        if (knn_matches2_1[i][0].distance < ratio_thresh * knn_matches2_1[i][1].distance) {
            good_matches2_1.push_back(knn_matches2_1[i][0]);
        }
    }

    // Extract location of good matches between img2 and img1
    std::vector<cv::Point2f> points2_1, points1_1;
    for (const auto &match : good_matches2_1) {
        points2_1.push_back(keypoints2[match.queryIdx].pt);
        points1_1.push_back(keypoints1[match.trainIdx].pt);
    }

    // Find the essential matrix between img2 and img1
    cv::Mat essential2_1 = cv::findEssentialMat(points2_1, points1_1, K);

    // Recover pose from the essential matrix between img2 and img1
    cv::Mat R2_1, t2_1;
    cv::recoverPose(essential2_1, points2_1, points1_1, K, R2_1, t2_1);

    // Convert rotation and translation to projection matrices
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F); // Projection matrix for img2 (reference)
    cv::Mat P2 = cv::Mat::eye(3, 4, CV_64F); // Projection matrix for img1

    // Set the projection matrix for img1 relative to img2
    R2_1.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    t2_1.copyTo(P2(cv::Rect(3, 0, 1, 3)));

    // Apply the camera intrinsic matrix K to both projection matrices
    P1 = K * P1;
    P2 = K * P2;

    // Triangulate points between img2 and img1
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, points1_1, points2_1, points4D);

    // Convert from homogeneous to 3D points and get colors from img2
    std::vector<cv::Point3d> points3D;
    std::vector<cv::Vec3b> colors;
    for (int i = 0; i < points4D.cols; ++i) {
        cv::Mat x = points4D.col(i);
        x /= x.at<float>(3); // Normalize to convert to 3D point
        points3D.push_back(cv::Point3d(x.at<float>(0), x.at<float>(1), x.at<float>(2)));
        cv::Point2f pt2 = points2_1[i];
        colors.push_back(img2_color.at<cv::Vec3b>(cv::Point(pt2.x, pt2.y)));
    }

    // Visualization
    cv::viz::Viz3d window("Coordinate Frame");
    window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

    // Add camera models to the visualization

    // Reference camera (img2)
    cv::Affine3d cam_pose2 = cv::Affine3d::Identity();
    cv::viz::WCameraPosition cam2(K, 0.5, cv::viz::Color::yellow());
    window.showWidget("Reference Camera (img2)", cam2, cam_pose2);

    // Camera pose for image 1 (relative to image 2)
    cv::Affine3d cam_pose1(R2_1, t2_1);
    cv::viz::WCameraPosition cam1(K, 0.5, cv::viz::Color::green());
    window.showWidget("Camera 1 (img1)", cam1, cam_pose1);

    // // Convert 3D points to a format suitable for viz::WCloud
    std::vector<cv::Vec3d> cloud;
    for (size_t i = 0; i < points3D.size(); ++i) {
        cloud.push_back(cv::Vec3d(points3D[i].x, points3D[i].y, points3D[i].z));
    }

    // Create the WCloud widget with colors
    cv::viz::WCloud cloud_widget(cloud, colors);

    // Show the point cloud
    window.showWidget("Triangulated Points", cloud_widget);

    // Display the 3D window until the user closes it
    window.spin();

    return 0;
}

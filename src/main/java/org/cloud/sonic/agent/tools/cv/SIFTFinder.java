/*
 *  Copyright (C) [SonicCloudOrg] Sonic Project
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package org.cloud.sonic.agent.tools.cv;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_features2d.FlannBasedMatcher;
import org.bytedeco.opencv.opencv_xfeatures2d.SIFT;
import org.cloud.sonic.agent.models.FindResult;
import org.cloud.sonic.agent.tools.file.UploadTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_features2d.drawMatchesKnn;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class SIFTFinder {
    private final Logger logger = LoggerFactory.getLogger(SIFTFinder.class);

    public FindResult getSIFTFindResult(File temFile, File beforeFile, org.openqa.selenium.Point assistedPoint) throws Exception {
        Mat image01 = imread(beforeFile.getAbsolutePath());
        Mat image02 = imread(temFile.getAbsolutePath());

        Mat image1 = new Mat();
        Mat image2 = new Mat();
        cvtColor(image01, image1, COLOR_BGR2GRAY);
        cvtColor(image02, image2, COLOR_BGR2GRAY);

        KeyPointVector keyPointVector1 = new KeyPointVector();
        KeyPointVector keyPointVector2 = new KeyPointVector();
        Mat image11 = new Mat();
        Mat image22 = new Mat();

        long start = System.currentTimeMillis();
        SIFT sift = SIFT.create();
        sift.detectAndCompute(image1, image1, keyPointVector1, image11);
        sift.detectAndCompute(image2, image2, keyPointVector2, image22);

        FlannBasedMatcher flannBasedMatcher = new FlannBasedMatcher();
        DMatchVectorVector matchPoints = new DMatchVectorVector();

        flannBasedMatcher.knnMatch(image11, image22, matchPoints, 2);
        logger.info("处理前共有匹配数：" + matchPoints.size());
        DMatchVectorVector goodMatches = new DMatchVectorVector();

        List<Integer> xs = new ArrayList<>();
        List<Integer> ys = new ArrayList<>();
        for (long i = 0; i < matchPoints.size(); i++) {
            if (matchPoints.get(i).size() >= 2) {
                DMatch match1 = matchPoints.get(i).get(0);
                DMatch match2 = matchPoints.get(i).get(1);

                if (match1.distance() <= 0.3 * match2.distance()) {
                    xs.add((int) keyPointVector1.get(match1.queryIdx()).pt().x());
                    ys.add((int) keyPointVector1.get(match1.queryIdx()).pt().y());
                    goodMatches.push_back(matchPoints.get(i));
                }
            }
        }
        logger.info("处理后匹配数：" + goodMatches.size());
        if (goodMatches.size() <= 4) {
            temFile.delete();
            beforeFile.delete();
            return null;
        }
        FindResult findResult = new FindResult();
        findResult.setTime((int) (System.currentTimeMillis() - start));
        Mat result = new Mat();

        drawMatchesKnn(image01, keyPointVector1, image02, keyPointVector2, goodMatches, result);

        //选取匹配到的坐标位置，如果有辅助控件的坐标，选取和辅助控件最相近的匹配坐标
        int resultX = 0;
        int resultY = 0;
        if(assistedPoint != null) {
            logger.info("辅助定位控件坐标为（" + assistedPoint.x + "," + assistedPoint.y + ")");
            circle(result, new Point(assistedPoint.x, assistedPoint.y), 5, Scalar.YELLOW, 10, CV_AA, 0);
            double distance = 10000;
            for(int i = 0; i < xs.size(); i++) {
                int x = xs.get(i);
                int y = ys.get(i);
                logger.info("坐标为（" + x + "," + y + ")");
                org.openqa.selenium.Point point = new org.openqa.selenium.Point(x, y);
                if(point.y >= assistedPoint.y) {
                    double d = point.y - assistedPoint.y;
                    logger.info("与辅助定位控件距离：" + d);
                    if(d < distance) {
                        distance = d;
                        resultX = x;
                        resultY = y;
                    }
                }
            }
        }
        if(resultX == 0 && resultY == 0) {
            resultX = majorityElement(xs);
            resultY = majorityElement(ys);
        }
        findResult.setX(resultX);
        findResult.setY(resultY);
        logger.info("结果坐标为（" + resultX + "," + resultY + ")");
        circle(result, new Point(resultX, resultY), 5, Scalar.RED, 10, CV_AA, 0);

        long time = Calendar.getInstance().getTimeInMillis();
        String fileName = "test-output" + File.separator + time + ".jpg";
        imwrite(fileName, result);
        findResult.setUrl(UploadTools.upload(new File(fileName), "imageFiles"));
        temFile.delete();
        beforeFile.delete();
        return findResult;
    }

    public static int majorityElement(List<Integer> nums) {
        double j;
        Collections.sort(nums);
        int size = nums.size();
        if (size % 2 == 1) {
            j = nums.get((size - 1) / 2);
        } else {
            j = (nums.get(size / 2 - 1) + nums.get(size / 2) + 0.0) / 2;
        }
        return (int) j;
    }
}

#!/usr/bin/env bash

source ./.venv/bin/activate

VOCAB_FILE="/opt/ORBSLAM2/Vocabulary/ORBvoc.txt"
EXAMPLES_FOLDER="/opt/ORBSLAM2/Examples"

# TUM examples
TUM_FOLDER=/media/john/Storage/datasets/TUM
for REPEAT in 1 2 3 4 5 6 7 8 9 10
do
    if [ ! -f ./trajectory-TUM-rgbd_dataset_frieburg1_xyz-mono-${REPEAT}.txt ]; then
        ./orbslam_mono_tum.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/Monocular/TUM1.yaml ${TUM_FOLDER}/rgbd_dataset_freiburg1_xyz
        mv ./trajectory.txt ./trajectory-TUM-rgbd_dataset_frieburg1_xyz-mono-${REPEAT}.txt
    fi

    if [ ! -f ./trajectory-TUM-rgbd_dataset_freiburg1_xyz-rgbd-${REPEAT}.txt ]; then
        ./orbslam_rgbd_tum.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/RGB-D/TUM1.yaml ${TUM_FOLDER}/rgbd_dataset_freiburg1_xyz ${EXAMPLES_FOLDER}/RGB-D/associations/fr1_xyz.txt
        mv ./trajectory.txt ./trajectory-TUM-rgbd_dataset_freiburg1_xyz-rgbd-${REPEAT}.txt
    fi

    if [ ! -f ./trajectory-TUM-rgbd_dataset_freiburg1_desk-mono-${REPEAT}.txt ]; then
        ./orbslam_mono_tum.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/Monocular/TUM1.yaml ${TUM_FOLDER}/rgbd_dataset_freiburg1_desk
        mv ./trajectory.txt ./trajectory-TUM-rgbd_dataset_freiburg1_desk-mono-${REPEAT}.txt
    fi

    if [ ! -f ./trajectory-TUM-rgbd_dataset_freiburg1_desk-rgbd-${REPEAT}.txt ]; then
        ./orbslam_rgbd_tum.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/RGB-D/TUM1.yaml ${TUM_FOLDER}/rgbd_dataset_freiburg1_desk ${EXAMPLES_FOLDER}/RGB-D/associations/fr1_desk.txt
        mv ./trajectory.txt ./trajectory-TUM-rgbd_dataset_freiburg1_desk-rgbd-${REPEAT}.txt
    fi
done

# EuRoC examples
EUROC_FOLDER=/media/john/Storage/datasets/EuRoC
for REPEAT in 1 2 3 4 5 6 7 8 9 10
do
    if [ ! -f ./trajectory-EuRoC-MH_01_easy-mono-${REPEAT}.txt ]; then
        ./orbslam_mono_euroc.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/Monocular/EuRoC.yaml ${EUROC_FOLDER}/MH_01_easy/cam0/data ${EXAMPLES_FOLDER}/Monocular/EuRoC_TimeStamps/MH01.txt
        mv ./trajectory.txt ./trajectory-EuRoC-MH_01_easy-mono-${REPEAT}.txt
    fi

    if [ ! -f ./trajectory-EuRoC-MH_01_easy-stereo-${REPEAT}.txt ]; then
        ./orbslam_stereo_euroc.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/Stereo/EuRoC.yaml ${EUROC_FOLDER}/MH_01_easy/cam0/data ${EUROC_FOLDER}/MH_01_easy/cam1/data ${EXAMPLES_FOLDER}/Stereo/EuRoC_TimeStamps/MH01.txt
        mv ./trajectory.txt ./trajectory-EuRoC-MH_01_easy-stereo-${REPEAT}.txt
    fi

    if [ ! -f ./trajectory-EuRoC-MH_04_difficult-mono-${REPEAT}.txt ]; then
        ./orbslam_mono_euroc.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/Monocular/EuRoC.yaml ${EUROC_FOLDER}/MH_04_difficult/cam0/data ${EXAMPLES_FOLDER}/Monocular/EuRoC_TimeStamps/MH04.txt
        mv ./trajectory.txt ./trajectory-EuRoC-MH_04_difficult-mono-${REPEAT}.txt
    fi

    if [ ! -f ./trajectory-EuRoC-MH_04_difficult-stereo-${REPEAT}.txt ]; then
        ./orbslam_stereo_euroc.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/Stereo/EuRoC.yaml ${EUROC_FOLDER}/MH_04_difficult/cam0/data ${EUROC_FOLDER}/MH_04_difficult/cam1/data ${EXAMPLES_FOLDER}/Stereo/EuRoC_TimeStamps/MH04.txt
        mv ./trajectory.txt ./trajectory-EuRoC-MH_04_difficult-stereo-${REPEAT}.txt
    fi
done



# EuRoC examples
KITTI_FOLDER=/media/john/Storage/datasets/KITTI/dataset/sequences
for REPEAT in 1 2 3 4 5 6 7 8 9 10
do
    if [ ! -f ./trajectory-KITTI-00-mono-${REPEAT}.txt ]; then
        ./orbslam_mono_kitti.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/Monocular/KITTI00-02.yaml ${KITTI_FOLDER}/00
        mv ./trajectory.txt ./trajectory-KITTI-00-mono-${REPEAT}.txt
    fi

    if [ ! -f ./trajectory-KITTI-00-stereo-${REPEAT}.txt ]; then
        ./orbslam_stereo_kitti.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/Stereo/KITTI00-02.yaml ${KITTI_FOLDER}/00
        mv ./trajectory.txt ./trajectory-KITTI-00-stereo-${REPEAT}.txt
    fi

    if [ ! -f ./trajectory-KITTI-03-mono-${REPEAT}.txt ]; then
        ./orbslam_mono_kitti.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/Monocular/KITTI03.yaml ${KITTI_FOLDER}/03
        mv ./trajectory.txt ./trajectory-KITTI-03-mono-${REPEAT}.txt
    fi

    if [ ! -f ./trajectory-KITTI-03-stereo-${REPEAT}.txt ]; then
        ./orbslam_stereo_kitti.py ${VOCAB_FILE} ${EXAMPLES_FOLDER}/Stereo/KITTI03.yaml ${KITTI_FOLDER}/03
        mv ./trajectory.txt ./trajectory-KITTI-03-stereo-${REPEAT}.txt
    fi
done

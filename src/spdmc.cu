// Sparse Dual Marching Cubes implementation
// Based on pdmc (Parallel Dual Marching Cubes) adapted for sparse voxels

#include "api.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

using V3f = float3;
using V4i = int4;
using Triangle = int3;

struct Quadric {
    float a2, ab, ac, ad;
    float b2, bc, bd;
    float c2, cd;
    float d2;

    __device__ Quadric() {
        a2 = ab = ac = ad = 0.f;
        b2 = bc = bd = 0.f;
        c2 = cd = 0.f;
        d2 = 0.f;
    }

    __device__ void add(const Quadric& other) {
        a2 += other.a2; ab += other.ab; ac += other.ac; ad += other.ad;
        b2 += other.b2; bc += other.bc; bd += other.bd;
        c2 += other.c2; cd += other.cd;
        d2 += other.d2;
    }
};

__device__ __constant__ int dmcEdgeToCorners[12][2] = {
    {0, 1}, {1, 3}, {2, 3}, {0, 2},
    {4, 5}, {5, 7}, {6, 7}, {4, 6},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}
};

// Dual Marching Cubes lookup tables from pdmc
__device__ __constant__ int dmcMcCorners[8][3] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 0},
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

// Edge locations: [x, y, z, axis] where axis: 0=X, 1=Y, 2=Z
__device__ __constant__ int dmcEdgeLocations[12][4] = {
    {0, 0, 0, 0},  // edge 0
    {1, 0, 0, 2},  // edge 1
    {0, 0, 1, 0},  // edge 2
    {0, 0, 0, 2},  // edge 3
    {0, 1, 0, 0},  // edge 4
    {1, 1, 0, 2},  // edge 5
    {0, 1, 1, 0},  // edge 6
    {0, 1, 0, 2},  // edge 7
    {0, 0, 0, 1},  // edge 8
    {1, 0, 0, 1},  // edge 9
    {1, 0, 1, 1},  // edge 10
    {0, 0, 1, 1},  // edge 11
};

// Patch indices for each MC case
__constant__ int mcFirstPatchIndex[257] = {
    0, 0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    21, 22, 24, 25, 28, 29, 31, 33, 35, 36, 38, 39, 41, 42, 43, 45, 46, 47, 49,
    51, 53, 54, 56, 59, 60, 61, 63, 65, 66, 67, 68, 69, 70, 71, 73, 74, 76, 77,
    79, 81, 82, 83, 85, 86, 87, 88, 89, 91, 93, 95, 96, 97, 99, 100, 102, 105, 107,
    109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 122, 123, 125, 127, 129, 130, 131, 132,
    133, 134, 136, 139, 141, 143, 145, 147, 149, 150, 153, 157, 159, 161, 163, 165, 166, 167, 168,
    169, 170, 171, 172, 173, 174, 175, 177, 179, 180, 181, 182, 183, 185, 186, 187, 189, 191, 193,
    195, 197, 200, 202, 203, 205, 206, 207, 208, 209, 210, 211, 213, 215, 218, 220, 223, 225, 229,
    231, 233, 235, 237, 238, 240, 241, 243, 244, 245, 247, 248, 249, 251, 253, 255, 256, 257, 259,
    260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 271, 272, 274, 275, 276, 277, 278, 279, 280,
    282, 283, 284, 285, 287, 289, 291, 292, 293, 295, 296, 297, 299, 300, 301, 302, 303, 304, 305,
    306, 307, 309, 310, 311, 312, 314, 315, 316, 317, 318, 320, 321, 322, 323, 324, 325, 327, 328,
    329, 330, 331, 332, 334, 335, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
    350, 351, 352, 353, 354, 355, 356, 357, 358, 358};

__constant__ int mcFirstEdgeIndex[359] = {
    0, 3, 6, 10, 13, 17, 20, 23, 28, 31, 34, 37, 41, 46, 50, 55,
    60, 64, 67, 71, 74, 77, 82, 85, 88, 93, 96, 99, 102, 108, 111, 114,
    118, 121, 125, 128, 134, 138, 141, 147, 152, 155, 160, 163, 166, 169, 173, 178,
    181, 184, 188, 191, 195, 198, 204, 207, 210, 213, 216, 219, 224, 230, 234, 237,
    242, 245, 251, 256, 260, 265, 270, 274, 277, 281, 287, 292, 295, 300, 303, 307,
    312, 315, 321, 326, 330, 334, 341, 348, 352, 355, 358, 361, 364, 367, 371, 374,
    378, 383, 386, 390, 396, 399, 402, 405, 408, 411, 415, 418, 423, 426, 431, 437,
    443, 448, 452, 457, 460, 464, 470, 475, 479, 482, 487, 492, 495, 499, 504, 507,
    511, 515, 522, 528, 533, 540, 544, 547, 550, 553, 556, 559, 563, 566, 571, 574,
    578, 581, 586, 589, 593, 597, 604, 607, 610, 613, 616, 619, 622, 625, 630, 633,
    639, 642, 647, 650, 656, 659, 666, 672, 677, 683, 689, 694, 700, 705, 712, 716,
    719, 724, 730, 733, 740, 746, 753, 759, 762, 765, 768, 771, 774, 777, 780, 783,
    787, 790, 793, 796, 800, 803, 806, 809, 812, 817, 820, 824, 827, 831, 836, 842,
    847, 853, 859, 864, 867, 870, 874, 877, 880, 883, 886, 891, 894, 897, 900, 903,
    908, 911, 914, 917, 920, 923, 929, 932, 936, 939, 943, 947, 952, 955, 962, 967,
    970, 977, 983, 986, 992, 996, 999, 1003, 1008, 1014, 1017, 1021, 1025, 1029, 1034, 1037,
    1044, 1049, 1052, 1057, 1061, 1066, 1072, 1079, 1084, 1088, 1093, 1099, 1105, 1110, 1113, 1118,
    1125, 1131, 1134, 1140, 1146, 1153, 1158, 1162, 1169, 1172, 1175, 1181, 1184, 1188, 1191, 1195,
    1198, 1202, 1206, 1210, 1215, 1221, 1224, 1229, 1236, 1241, 1244, 1249, 1255, 1262, 1266, 1271,
    1276, 1280, 1285, 1291, 1294, 1299, 1306, 1312, 1317, 1320, 1326, 1332, 1338, 1345, 1352, 1355,
    1358, 1363, 1367, 1373, 1376, 1381, 1384, 1389, 1395, 1402, 1408, 1415, 1422, 1425, 1428, 1434,
    1437, 1443, 1448, 1454, 1459, 1465, 1469, 1472, 1476, 1481, 1486, 1490, 1495, 1499, 1505, 1508,
    1513, 1519, 1523, 1526, 1530, 1533, 1536};

__constant__ int mcEdgeIndex[1536] = {
    0, 3, 8, 0, 1, 9, 1, 3, 8, 9, 4, 7, 8, 0, 3, 4, 7, 0, 1, 9, 4, 7, 8, 1,
    3, 4, 7, 9, 4, 5, 9, 0, 3, 8, 4, 5, 9, 0, 1, 4, 5, 1, 3, 4, 5, 8, 5, 7,
    8, 9, 0, 3, 5, 7, 9, 0, 1, 5, 7, 8, 1, 3, 5, 7, 2, 3, 11, 0, 2, 8, 11, 0,
    1, 9, 2, 3, 11, 1, 2, 8, 9, 11, 4, 7, 8, 2, 3, 11, 0, 2, 4, 7, 11, 0, 1, 9,
    4, 7, 8, 2, 3, 11, 1, 2, 4, 7, 9, 11, 4, 5, 9, 2, 3, 11, 0, 2, 8, 11, 4, 5,
    9, 0, 1, 4, 5, 2, 3, 11, 1, 2, 4, 5, 8, 11, 5, 7, 8, 9, 2, 3, 11, 0, 2, 5,
    7, 9, 11, 0, 1, 5, 7, 8, 2, 3, 11, 1, 2, 5, 7, 11, 1, 2, 10, 0, 3, 8, 1, 2,
    10, 0, 2, 9, 10, 2, 3, 8, 9, 10, 4, 7, 8, 1, 2, 10, 0, 3, 4, 7, 1, 2, 10, 0,
    2, 9, 10, 4, 7, 8, 2, 3, 4, 7, 9, 10, 4, 5, 9, 1, 2, 10, 0, 3, 8, 4, 5, 9,
    1, 2, 10, 0, 2, 4, 5, 10, 2, 3, 4, 5, 8, 10, 5, 7, 8, 9, 1, 2, 10, 0, 3, 5,
    7, 9, 1, 2, 10, 0, 2, 5, 7, 8, 10, 2, 3, 5, 7, 10, 1, 3, 10, 11, 0, 1, 8, 10,
    11, 0, 3, 9, 10, 11, 8, 9, 10, 11, 4, 7, 8, 1, 3, 10, 11, 0, 1, 4, 7, 10, 11, 0,
    3, 9, 10, 11, 4, 7, 8, 4, 7, 9, 10, 11, 4, 5, 9, 1, 3, 10, 11, 0, 1, 8, 10, 11,
    4, 5, 9, 0, 3, 4, 5, 10, 11, 4, 5, 8, 10, 11, 5, 7, 8, 9, 1, 3, 10, 11, 0, 1,
    5, 7, 9, 10, 11, 0, 3, 5, 7, 8, 10, 11, 5, 7, 10, 11, 6, 7, 11, 0, 3, 8, 6, 7,
    11, 0, 1, 9, 6, 7, 11, 1, 3, 8, 9, 6, 7, 11, 4, 6, 8, 11, 0, 3, 4, 6, 11, 0,
    1, 9, 4, 6, 8, 11, 1, 3, 4, 6, 9, 11, 4, 5, 9, 6, 7, 11, 0, 3, 8, 4, 5, 9,
    6, 7, 11, 0, 1, 4, 5, 6, 7, 11, 1, 3, 4, 5, 8, 6, 7, 11, 5, 6, 8, 9, 11, 0,
    3, 5, 6, 9, 11, 0, 1, 5, 6, 8, 11, 1, 3, 5, 6, 11, 2, 3, 6, 7, 0, 2, 6, 7,
    8, 0, 1, 9, 2, 3, 6, 7, 1, 2, 6, 7, 8, 9, 2, 3, 4, 6, 8, 0, 2, 4, 6, 0,
    1, 9, 2, 3, 4, 6, 8, 1, 2, 4, 6, 9, 4, 5, 9, 2, 3, 6, 7, 0, 2, 6, 7, 8,
    4, 5, 9, 0, 1, 4, 5, 2, 3, 6, 7, 1, 2, 4, 5, 6, 7, 8, 2, 3, 5, 6, 8, 9,
    0, 2, 5, 6, 9, 0, 1, 2, 3, 5, 6, 8, 1, 2, 5, 6, 1, 2, 10, 6, 7, 11, 0, 3,
    8, 1, 2, 10, 6, 7, 11, 0, 2, 9, 10, 6, 7, 11, 2, 3, 8, 9, 10, 6, 7, 11, 4, 6,
    8, 11, 1, 2, 10, 0, 3, 4, 6, 11, 1, 2, 10, 0, 2, 9, 10, 4, 6, 8, 11, 2, 3, 4,
    6, 9, 10, 11, 4, 5, 9, 1, 2, 10, 6, 7, 11, 0, 3, 8, 4, 5, 9, 1, 2, 10, 6, 7,
    11, 0, 2, 4, 5, 10, 6, 7, 11, 2, 3, 4, 5, 8, 10, 6, 7, 11, 5, 6, 8, 9, 11, 1,
    2, 10, 0, 3, 5, 6, 9, 11, 1, 2, 10, 0, 2, 5, 6, 8, 10, 11, 2, 3, 5, 6, 10, 11,
    1, 3, 6, 7, 10, 0, 1, 6, 7, 8, 10, 0, 3, 6, 7, 9, 10, 6, 7, 8, 9, 10, 1, 3,
    4, 6, 8, 10, 0, 1, 4, 6, 10, 0, 3, 4, 6, 8, 9, 10, 4, 6, 9, 10, 4, 5, 9, 1,
    3, 6, 7, 10, 0, 1, 6, 7, 8, 10, 4, 5, 9, 0, 3, 4, 5, 6, 7, 10, 4, 5, 6, 7,
    8, 10, 1, 3, 5, 6, 8, 9, 10, 0, 1, 5, 6, 9, 10, 0, 3, 8, 5, 6, 10, 5, 6, 10,
    5, 6, 10, 0, 3, 8, 5, 6, 10, 0, 1, 9, 5, 6, 10, 1, 3, 8, 9, 5, 6, 10, 4, 7,
    8, 5, 6, 10, 0, 3, 4, 7, 5, 6, 10, 0, 1, 9, 4, 7, 8, 5, 6, 10, 1, 3, 4, 7,
    9, 5, 6, 10, 4, 6, 9, 10, 0, 3, 8, 4, 6, 9, 10, 0, 1, 4, 6, 10, 1, 3, 4, 6,
    8, 10, 6, 7, 8, 9, 10, 0, 3, 6, 7, 9, 10, 0, 1, 6, 7, 8, 10, 1, 3, 6, 7, 10,
    2, 3, 11, 5, 6, 10, 0, 2, 8, 11, 5, 6, 10, 0, 1, 9, 2, 3, 11, 5, 6, 10, 1, 2,
    8, 9, 11, 5, 6, 10, 4, 7, 8, 2, 3, 11, 5, 6, 10, 0, 2, 4, 7, 11, 5, 6, 10, 0,
    1, 9, 4, 7, 8, 2, 3, 11, 5, 6, 10, 1, 2, 4, 7, 9, 11, 5, 6, 10, 4, 6, 9, 10,
    2, 3, 11, 0, 2, 8, 11, 4, 6, 9, 10, 0, 1, 4, 6, 10, 2, 3, 11, 1, 2, 4, 6, 8,
    10, 11, 6, 7, 8, 9, 10, 2, 3, 11, 0, 2, 6, 7, 9, 10, 11, 0, 1, 6, 7, 8, 10, 2,
    3, 11, 1, 2, 6, 7, 10, 11, 1, 2, 5, 6, 0, 3, 8, 1, 2, 5, 6, 0, 2, 5, 6, 9,
    2, 3, 5, 6, 8, 9, 4, 7, 8, 1, 2, 5, 6, 0, 3, 4, 7, 1, 2, 5, 6, 0, 2, 5,
    6, 9, 4, 7, 8, 2, 3, 4, 5, 6, 7, 9, 1, 2, 4, 6, 9, 0, 3, 8, 1, 2, 4, 6,
    9, 0, 2, 4, 6, 2, 3, 4, 6, 8, 1, 2, 6, 7, 8, 9, 0, 1, 2, 3, 6, 7, 9, 0,
    2, 6, 7, 8, 2, 3, 6, 7, 1, 3, 5, 6, 11, 0, 1, 5, 6, 8, 11, 0, 3, 5, 6, 9,
    11, 5, 6, 8, 9, 11, 4, 7, 8, 1, 3, 5, 6, 11, 0, 1, 4, 5, 6, 7, 11, 0, 3, 5,
    6, 9, 11, 4, 7, 8, 4, 5, 6, 7, 9, 11, 1, 3, 4, 6, 9, 11, 0, 1, 4, 6, 8, 9,
    11, 0, 3, 4, 6, 11, 4, 6, 8, 11, 1, 3, 6, 7, 8, 9, 11, 0, 1, 9, 6, 7, 11, 0,
    3, 6, 7, 8, 11, 6, 7, 11, 5, 7, 10, 11, 0, 3, 8, 5, 7, 10, 11, 0, 1, 9, 5, 7,
    10, 11, 1, 3, 8, 9, 5, 7, 10, 11, 4, 5, 8, 10, 11, 0, 3, 4, 5, 10, 11, 0, 1, 9,
    4, 5, 8, 10, 11, 1, 3, 4, 5, 9, 10, 11, 4, 7, 9, 10, 11, 0, 3, 8, 4, 7, 9, 10,
    11, 0, 1, 4, 7, 10, 11, 1, 3, 4, 7, 8, 10, 11, 8, 9, 10, 11, 0, 3, 9, 10, 11, 0,
    1, 8, 10, 11, 1, 3, 10, 11, 2, 3, 5, 7, 10, 0, 2, 5, 7, 8, 10, 0, 1, 9, 2, 3,
    5, 7, 10, 1, 2, 5, 7, 8, 9, 10, 2, 3, 4, 5, 8, 10, 0, 2, 4, 5, 10, 0, 1, 9,
    2, 3, 4, 5, 8, 10, 1, 2, 4, 5, 9, 10, 2, 3, 4, 7, 9, 10, 0, 2, 4, 7, 8, 9,
    10, 0, 1, 2, 3, 4, 7, 10, 4, 7, 8, 1, 2, 10, 2, 3, 8, 9, 10, 0, 2, 9, 10, 0,
    1, 2, 3, 8, 10, 1, 2, 10, 1, 2, 5, 7, 11, 0, 3, 8, 1, 2, 5, 7, 11, 0, 2, 5,
    7, 9, 11, 2, 3, 5, 7, 8, 9, 11, 1, 2, 4, 5, 8, 11, 0, 1, 2, 3, 4, 5, 11, 0,
    2, 4, 5, 8, 9, 11, 4, 5, 9, 2, 3, 11, 1, 2, 4, 7, 9, 11, 0, 3, 8, 1, 2, 4,
    7, 9, 11, 0, 2, 4, 7, 11, 2, 3, 4, 7, 8, 11, 1, 2, 8, 9, 11, 0, 1, 2, 3, 9,
    11, 0, 2, 8, 11, 2, 3, 11, 1, 3, 5, 7, 0, 1, 5, 7, 8, 0, 3, 5, 7, 9, 5, 7,
    8, 9, 1, 3, 4, 5, 8, 0, 1, 4, 5, 0, 3, 4, 5, 8, 9, 4, 5, 9, 1, 3, 4, 7,
    9, 0, 1, 4, 7, 8, 9, 0, 3, 4, 7, 4, 7, 8, 1, 3, 8, 9, 0, 1, 9, 0, 3, 8};

// dmcEdgeOffset table
__constant__ int8_t const dmcEdgeOffset[256][12] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1},
    {0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1},
    {-1, 0, -1, 0, -1, -1, -1, -1, 0, 0, -1, -1},
    {-1, -1, -1, -1, 0, -1, -1, 0, 0, -1, -1, -1},
    {0, -1, -1, 0, 0, -1, -1, 0, -1, -1, -1, -1},
    {0, 0, -1, -1, 1, -1, -1, 1, 1, 0, -1, -1},
    {-1, 0, -1, 0, 0, -1, -1, 0, -1, 0, -1, -1},
    {-1, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1},
    {0, -1, -1, 0, 1, 1, -1, -1, 0, 1, -1, -1},
    {0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1},
    {-1, 0, -1, 0, 0, 0, -1, -1, 0, -1, -1, -1},
    {-1, -1, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1},
    {0, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1},
    {0, 0, -1, -1, -1, 0, -1, 0, 0, -1, -1, -1},
    {-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1},
    {-1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0},
    {0, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, 0},
    {0, 0, 1, 1, -1, -1, -1, -1, -1, 0, -1, 1},
    {-1, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, 0},
    {-1, -1, 1, 1, 0, -1, -1, 0, 0, -1, -1, 1},
    {0, -1, 0, -1, 0, -1, -1, 0, -1, -1, -1, 0},
    {0, 0, 2, 2, 1, -1, -1, 1, 1, 0, -1, 2},
    {-1, 0, 0, -1, 0, -1, -1, 0, -1, 0, -1, 0},
    {-1, -1, 1, 1, 0, 0, -1, -1, -1, 0, -1, 1},
    {0, -1, 0, -1, 1, 1, -1, -1, 0, 1, -1, 0},
    {0, 0, 1, 1, 0, 0, -1, -1, -1, -1, -1, 1},
    {-1, 0, 0, -1, 0, 0, -1, -1, 0, -1, -1, 0},
    {-1, -1, 1, 1, -1, 0, -1, 0, 0, 0, -1, 1},
    {0, -1, 0, -1, -1, 0, -1, 0, -1, 0, -1, 0},
    {0, 0, 1, 1, -1, 0, -1, 0, 0, -1, -1, 1},
    {-1, 0, 0, -1, -1, 0, -1, 0, -1, -1, -1, 0},
    {-1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1},
    {0, 1, 1, 0, -1, -1, -1, -1, 0, -1, 1, -1},
    {0, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1},
    {-1, -1, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1},
    {-1, 1, 1, -1, 0, -1, -1, 0, 0, -1, 1, -1},
    {0, 1, 1, 0, 0, -1, -1, 0, -1, -1, 1, -1},
    {0, -1, 0, -1, 1, -1, -1, 1, 1, 0, 0, -1},
    {-1, -1, 0, 0, 0, -1, -1, 0, -1, 0, 0, -1},
    {-1, 1, 1, -1, 0, 0, -1, -1, -1, 0, 1, -1},
    {0, 2, 2, 0, 1, 1, -1, -1, 0, 1, 2, -1},
    {0, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0, -1},
    {-1, -1, 0, 0, 0, 0, -1, -1, 0, -1, 0, -1},
    {-1, 1, 1, -1, -1, 0, -1, 0, 0, 0, 1, -1},
    {0, 1, 1, 0, -1, 0, -1, 0, -1, 0, 1, -1},
    {0, -1, 0, -1, -1, 0, -1, 0, 0, -1, 0, -1},
    {-1, -1, 0, 0, -1, 0, -1, 0, -1, -1, 0, -1},
    {-1, 0, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0},
    {0, 0, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0},
    {0, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0},
    {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0},
    {-1, 1, -1, 1, 0, -1, -1, 0, 0, -1, 1, 1},
    {0, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, 0},
    {0, -1, -1, 0, 1, -1, -1, 1, 1, 0, 0, 0},
    {-1, -1, -1, -1, 0, -1, -1, 0, -1, 0, 0, 0},
    {-1, 1, -1, 1, 0, 0, -1, -1, -1, 0, 1, 1},
    {0, 0, -1, -1, 1, 1, -1, -1, 0, 1, 0, 0},
    {0, -1, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0},
    {-1, -1, -1, -1, 0, 0, -1, -1, 0, -1, 0, 0},
    {-1, 1, -1, 1, -1, 0, -1, 0, 0, 0, 1, 1},
    {0, 0, -1, -1, -1, 0, -1, 0, -1, 0, 0, 0},
    {0, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0},
    {-1, -1, -1, -1, -1, 0, -1, 0, -1, -1, 0, 0},
    {-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0},
    {0, -1, -1, 0, -1, -1, 1, 1, 0, -1, -1, 1},
    {0, 0, -1, -1, -1, -1, 1, 1, -1, 0, -1, 1},
    {-1, 0, -1, 0, -1, -1, 1, 1, 0, 0, -1, 1},
    {-1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, 0},
    {0, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1, 0},
    {0, 0, -1, -1, 1, -1, 1, -1, 1, 0, -1, 1},
    {-1, 0, -1, 0, 0, -1, 0, -1, -1, 0, -1, 0},
    {-1, -1, -1, -1, 0, 0, 1, 1, -1, 0, -1, 1},
    {0, -1, -1, 0, 1, 1, 2, 2, 0, 1, -1, 2},
    {0, 0, -1, -1, 0, 0, 1, 1, -1, -1, -1, 1},
    {-1, 0, -1, 0, 0, 0, 1, 1, 0, -1, -1, 1},
    {-1, -1, -1, -1, -1, 0, 0, -1, 0, 0, -1, 0},
    {0, -1, -1, 0, -1, 0, 0, -1, -1, 0, -1, 0},
    {0, 0, -1, -1, -1, 0, 0, -1, 0, -1, -1, 0},
    {-1, 0, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0},
    {-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1},
    {0, -1, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1},
    {0, 0, 1, 1, -1, -1, 1, 1, -1, 0, -1, -1},
    {-1, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1},
    {-1, -1, 0, 0, 0, -1, 0, -1, 0, -1, -1, -1},
    {0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1},
    {0, 0, 1, 1, 1, -1, 1, -1, 1, 0, -1, -1},
    {-1, 0, 0, -1, 0, -1, 0, -1, -1, 0, -1, -1},
    {-1, -1, 1, 1, 0, 0, 1, 1, -1, 0, -1, -1},
    {0, -1, 0, -1, 1, 1, 0, 0, 0, 1, -1, -1},
    {0, 0, 1, 1, 0, 0, 1, 1, -1, -1, -1, -1},
    {-1, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1, -1},
    {-1, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, -1},
    {0, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1, -1},
    {0, 0, 0, 0, -1, 0, 0, -1, 0, -1, -1, -1},
    {-1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1},
    {-1, 0, 0, -1, -1, -1, 1, 1, -1, -1, 0, 1},
    {0, 1, 1, 0, -1, -1, 2, 2, 0, -1, 1, 2},
    {0, -1, 0, -1, -1, -1, 1, 1, -1, 0, 0, 1},
    {-1, -1, 0, 0, -1, -1, 1, 1, 0, 0, 0, 1},
    {-1, 1, 1, -1, 0, -1, 0, -1, 0, -1, 1, 0},
    {0, 1, 1, 0, 0, -1, 0, -1, -1, -1, 1, 0},
    {0, -1, 0, -1, 1, -1, 1, -1, 1, 0, 0, 1},
    {-1, -1, 0, 0, 0, -1, 0, -1, -1, 0, 0, 0},
    {-1, 1, 1, -1, 0, 0, 2, 2, -1, 0, 1, 2},
    {0, 2, 2, 0, 1, 1, 3, 3, 0, 1, 2, 3},
    {0, -1, 0, -1, 0, 0, 1, 1, -1, -1, 0, 1},
    {-1, -1, 0, 0, 0, 0, 1, 1, 0, -1, 0, 1},
    {-1, 1, 1, -1, -1, 0, 0, -1, 0, 0, 1, 0},
    {0, 1, 1, 0, -1, 0, 0, -1, -1, 0, 1, 0},
    {0, -1, 0, -1, -1, 0, 0, -1, 0, -1, 0, 0},
    {-1, -1, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0},
    {-1, 0, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1},
    {0, 0, -1, -1, -1, -1, 0, 0, 0, -1, 0, -1},
    {0, -1, -1, 0, -1, -1, 0, 0, -1, 0, 0, -1},
    {-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1},
    {-1, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0, -1},
    {0, 0, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1},
    {0, -1, -1, 0, 0, -1, 0, -1, 0, 0, 0, -1},
    {-1, -1, -1, -1, 0, -1, 0, -1, -1, 0, 0, -1},
    {-1, 1, -1, 1, 0, 0, 1, 1, -1, 0, 1, -1},
    {0, 0, -1, -1, 1, 1, 0, 0, 0, 1, 0, -1},
    {0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0, -1},
    {-1, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, -1},
    {-1, 0, -1, 0, -1, 0, 0, -1, 0, 0, 0, -1},
    {0, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0, -1},
    {0, -1, -1, 0, -1, 1, 1, -1, 0, -1, 1, -1},
    {-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1},
    {-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1},
    {0, -1, -1, 0, -1, 1, 1, -1, 0, -1, 1, -1},
    {0, 0, -1, -1, -1, 1, 1, -1, -1, 0, 1, -1},
    {-1, 0, -1, 0, -1, 1, 1, -1, 0, 0, 1, -1},
    {-1, -1, -1, -1, 0, 1, 1, 0, 0, -1, 1, -1},
    {0, -1, -1, 0, 0, 1, 1, 0, -1, -1, 1, -1},
    {0, 0, -1, -1, 1, 2, 2, 1, 1, 0, 2, -1},
    {-1, 0, -1, 0, 0, 1, 1, 0, -1, 0, 1, -1},
    {-1, -1, -1, -1, 0, -1, 0, -1, -1, 0, 0, -1},
    {0, -1, -1, 0, 1, -1, 1, -1, 0, 1, 1, -1},
    {0, 0, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1},
    {-1, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0, -1},
    {-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1},
    {0, -1, -1, 0, -1, -1, 0, 0, -1, 0, 0, -1},
    {0, 0, -1, -1, -1, -1, 0, 0, 0, -1, 0, -1},
    {-1, 0, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1},
    {-1, -1, 0, 0, -1, 1, 1, -1, -1, -1, 1, 0},
    {0, -1, 0, -1, -1, 1, 1, -1, 0, -1, 1, 0},
    {0, 0, 1, 1, -1, 2, 2, -1, -1, 0, 2, 1},
    {-1, 0, 0, -1, -1, 1, 1, -1, 0, 0, 1, 0},
    {-1, -1, 1, 1, 0, 2, 2, 0, 0, -1, 2, 1},
    {0, -1, 0, -1, 0, 1, 1, 0, -1, -1, 1, 0},
    {0, 0, 2, 2, 1, 3, 3, 1, 1, 0, 3, 2},
    {-1, 0, 0, -1, 0, 1, 1, 0, -1, 0, 1, 0},
    {-1, -1, 1, 1, 0, -1, 0, -1, -1, 0, 0, 1},
    {0, -1, 0, -1, 1, -1, 1, -1, 0, 1, 1, 0},
    {0, 0, 1, 1, 0, -1, 0, -1, -1, -1, 0, 1},
    {-1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0},
    {-1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 1},
    {0, -1, 0, -1, -1, -1, 0, 0, -1, 0, 0, 0},
    {0, 0, 1, 1, -1, -1, 0, 0, 0, -1, 0, 1},
    {-1, 0, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0},
    {-1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1},
    {0, 1, 1, 0, -1, 1, 1, -1, 0, -1, -1, -1},
    {0, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1, -1},
    {-1, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, -1},
    {-1, 1, 1, -1, 0, 1, 1, 0, 0, -1, -1, -1},
    {0, 1, 1, 0, 0, 1, 1, 0, -1, -1, -1, -1},
    {0, -1, 0, -1, 1, 0, 0, 1, 1, 0, -1, -1},
    {-1, -1, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1},
    {-1, 0, 0, -1, 0, -1, 0, -1, -1, 0, -1, -1},
    {0, 1, 1, 0, 1, -1, 1, -1, 0, 1, -1, -1},
    {0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1},
    {-1, -1, 0, 0, 0, -1, 0, -1, 0, -1, -1, -1},
    {-1, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1},
    {0, 0, 0, 0, -1, -1, 0, 0, -1, 0, -1, -1},
    {0, -1, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1},
    {-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1},
    {-1, 0, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0},
    {0, 0, -1, -1, -1, 0, 0, -1, 0, -1, -1, 0},
    {0, -1, -1, 0, -1, 0, 0, -1, -1, 0, -1, 0},
    {-1, -1, -1, -1, -1, 0, 0, -1, 0, 0, -1, 0},
    {-1, 1, -1, 1, 0, 1, 1, 0, 0, -1, -1, 1},
    {0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0},
    {0, -1, -1, 0, 1, 0, 0, 1, 1, 0, -1, 0},
    {-1, -1, -1, -1, 0, 0, 0, 0, -1, 0, -1, 0},
    {-1, 0, -1, 0, 0, -1, 0, -1, -1, 0, -1, 0},
    {0, 0, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0},
    {0, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1, 0},
    {-1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, 0},
    {-1, 0, -1, 0, -1, -1, 0, 0, 0, 0, -1, 0},
    {0, 0, -1, -1, -1, -1, 1, 1, -1, 0, -1, 1},
    {0, -1, -1, 0, -1, -1, 0, 0, 0, -1, -1, 0},
    {-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0},
    {-1, -1, -1, -1, -1, 0, -1, 0, -1, -1, 0, 0},
    {0, -1, -1, 0, -1, 1, -1, 1, 0, -1, 1, 1},
    {0, 0, -1, -1, -1, 1, -1, 1, -1, 0, 1, 1},
    {-1, 0, -1, 0, -1, 1, -1, 1, 0, 0, 1, 1},
    {-1, -1, -1, -1, 0, 0, -1, -1, 0, -1, 0, 0},
    {0, -1, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0},
    {0, 0, -1, -1, 1, 1, -1, -1, 1, 0, 1, 1},
    {-1, 0, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0},
    {-1, -1, -1, -1, 0, -1, -1, 0, -1, 0, 0, 0},
    {0, -1, -1, 0, 1, -1, -1, 1, 0, 1, 1, 1},
    {0, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, 0},
    {-1, 0, -1, 0, 0, -1, -1, 0, 0, -1, 0, 0},
    {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0},
    {0, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0},
    {0, 0, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0},
    {-1, 0, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0},
    {-1, -1, 0, 0, -1, 0, -1, 0, -1, -1, 0, -1},
    {0, -1, 0, -1, -1, 0, -1, 0, 0, -1, 0, -1},
    {0, 0, 1, 1, -1, 1, -1, 1, -1, 0, 1, -1},
    {-1, 0, 0, -1, -1, 0, -1, 0, 0, 0, 0, -1},
    {-1, -1, 0, 0, 0, 0, -1, -1, 0, -1, 0, -1},
    {0, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0, -1},
    {0, 0, 1, 1, 1, 1, -1, -1, 1, 0, 1, -1},
    {-1, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0, -1},
    {-1, -1, 0, 0, 0, -1, -1, 0, -1, 0, 0, -1},
    {0, -1, 0, -1, 0, -1, -1, 0, 0, 0, 0, -1},
    {0, 0, 0, 0, 0, -1, -1, 0, -1, -1, 0, -1},
    {-1, 1, 1, -1, 0, -1, -1, 0, 0, -1, 1, -1},
    {-1, -1, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1},
    {0, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1},
    {0, 0, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1},
    {-1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1},
    {-1, 0, 0, -1, -1, 0, -1, 0, -1, -1, -1, 0},
    {0, 1, 1, 0, -1, 1, -1, 1, 0, -1, -1, 1},
    {0, -1, 0, -1, -1, 0, -1, 0, -1, 0, -1, 0},
    {-1, -1, 0, 0, -1, 0, -1, 0, 0, 0, -1, 0},
    {-1, 0, 0, -1, 0, 0, -1, -1, 0, -1, -1, 0},
    {0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0},
    {0, -1, 0, -1, 0, 0, -1, -1, 0, 0, -1, 0},
    {-1, -1, 1, 1, 0, 0, -1, -1, -1, 0, -1, 1},
    {-1, 0, 0, -1, 0, -1, -1, 0, -1, 0, -1, 0},
    {0, 1, 1, 0, 1, -1, -1, 1, 0, 1, -1, 1},
    {0, -1, 0, -1, 0, -1, -1, 0, -1, -1, -1, 0},
    {-1, -1, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0},
    {-1, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, 0},
    {0, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0},
    {0, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, 0},
    {-1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0},
    {-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1},
    {0, 0, -1, -1, -1, 0, -1, 0, 0, -1, -1, -1},
    {0, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1},
    {-1, -1, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1},
    {-1, 0, -1, 0, 0, 0, -1, -1, 0, -1, -1, -1},
    {0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1},
    {0, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1, -1},
    {-1, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1},
    {-1, 0, -1, 0, 0, -1, -1, 0, -1, 0, -1, -1},
    {0, 0, -1, -1, 0, -1, -1, 0, 0, 0, -1, -1},
    {0, -1, -1, 0, 0, -1, -1, 0, -1, -1, -1, -1},
    {-1, -1, -1, -1, 0, -1, -1, 0, 0, -1, -1, -1},
    {-1, 0, -1, 0, -1, -1, -1, -1, 0, 0, -1, -1},
    {0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1},
    {0, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
};

// dmcQuad table: [edge_type][4_vertices][dx, dy, dz, edge_id]
__constant__ int const dmcQuad[6][4][4] = {
    {{0, 0, 0, 0}, {0, -1, 0, 4}, {0, -1, -1, 6}, {0, 0, -1, 2}},
    {{0, 0, 0, 8}, {0, 0, -1, 11}, {-1, 0, -1, 10}, {-1, 0, 0, 9}},
    {{0, 0, 0, 3}, {-1, 0, 0, 1}, {-1, -1, 0, 5}, {0, -1, 0, 7}},

    {{0, 0, 0, 0}, {0, 0, -1, 2}, {0, -1, -1, 6}, {0, -1, 0, 4}},
    {{0, 0, 0, 8}, {-1, 0, 0, 9}, {-1, 0, -1, 10}, {0, 0, -1, 11}},
    {{0, 0, 0, 3}, {0, -1, 0, 7}, {-1, -1, 0, 5}, {-1, 0, 0, 1}},
};

// Edge masks for each marching cubes case
__device__ __constant__ int dmcEdgeTable[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

__device__ __constant__ int3 dmcCornerOffset[8] = {
    {0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
    {0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}
};

__device__ float sigmoidAdjust(float t) {
    constexpr float BETA = 5.0f;
    float centered = t - 0.5f;
    return 1.0f / (1.0f + expf(-BETA * centered));
}

__global__ void calculatePatchQuadricsKernel(const int* coords, const float* corners, int N, float iso,
                                             const uint8_t* cellCodes, const int* patchPrefix,
                                             Quadric* patchQuadrics, V3f* patchCentroids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint8_t cubeCode = cellCodes[i];
    if (cubeCode == 0 || cubeCode == 255) return;

    int vx = coords[i*3+0];
    int vy = coords[i*3+1];
    int vz = coords[i*3+2];

    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = corners[i*8 + j];
    }

    int firstPatch = mcFirstPatchIndex[cubeCode];
    int lastPatch = mcFirstPatchIndex[cubeCode + 1];
    int outputBaseIdx = patchPrefix[i];

    for (int patch_index = firstPatch; patch_index < lastPatch; ++patch_index) {
        Quadric q_accum; // patch quadratic
        V3f centroid = make_float3(0.0f, 0.0f, 0.0f);
        float num_points = 0.0f;

        int firstEdge = mcFirstEdgeIndex[patch_index];
        int lastEdge = mcFirstEdgeIndex[patch_index + 1];

        for (int edge_index = firstEdge; edge_index < lastEdge; ++edge_index) {
            int eid = mcEdgeIndex[edge_index];

            // MC
            int c0_idx = dmcEdgeToCorners[eid][0];
            int c1_idx = dmcEdgeToCorners[eid][1];

            float v0_val = v[c0_idx];
            float v1_val = v[c1_idx];
            float t = (iso - v0_val) / (v1_val - v0_val + 1e-30f);

            t = sigmoidAdjust(t);
            t = fminf(fmaxf(t, 0.f), 1.f);

            int3 off0 = dmcCornerOffset[c0_idx];
            int3 off1 = dmcCornerOffset[c1_idx];

            V3f p = make_float3(vx + off0.x + t * (off1.x - off0.x),
                                vy + off0.y + t * (off1.y - off0.y),
                                vz + off0.z + t * (off1.z - off0.z));

            // Accumulate for centroid
            centroid.x += p.x;
            centroid.y += p.y;
            centroid.z += p.z;
            num_points += 1.0f;

            // Calculate gradient at edge point using trilinear interpolation
            // Point p is in local coordinates [vx, vx+1] x [vy, vy+1] x [vz, vz+1]
            float local_x = p.x - vx;
            float local_y = p.y - vy;
            float local_z = p.z - vz;

            // Trilinear interpolation of gradient
            // Gradient in x: derivative with respect to x
            // float grad_x = (1.0f - local_y) * (1.0f - local_z) * (v[1] - v[0]) +
            //                local_y * (1.0f - local_z) * (v[3] - v[2]) +
            //                (1.0f - local_y) * local_z * (v[5] - v[4]) +
            //                local_y * local_z * (v[7] - v[6]);

            // float grad_y = (1.0f - local_x) * (1.0f - local_z) * (v[2] - v[0]) +
            //                local_x * (1.0f - local_z) * (v[3] - v[1]) +
            //                (1.0f - local_x) * local_z * (v[6] - v[4]) +
            //                local_x * local_z * (v[7] - v[5]);

            // float grad_z = (1.0f - local_x) * (1.0f - local_y) * (v[4] - v[0]) +
            //                local_x * (1.0f - local_y) * (v[5] - v[1]) +
            //                (1.0f - local_x) * local_y * (v[6] - v[2]) +
            //                local_x * local_y * (v[7] - v[3]);

            // Gradient in x
            float gx0 = (v[1] - v[0]) * (1 - local_y) * (1 - local_z) +
                        (v[3] - v[2]) * local_y * (1 - local_z) +
                        (v[5] - v[4]) * (1 - local_y) * local_z +
                        (v[7] - v[6]) * local_y * local_z;

            // Gradient in y
            float gy0 = (v[2] - v[0]) * (1 - local_x) * (1 - local_z) +
                        (v[3] - v[1]) * local_x * (1 - local_z) +
                        (v[6] - v[4]) * (1 - local_x) * local_z +
                        (v[7] - v[5]) * local_x * local_z;

            // Gradient in z
            float gz0 = (v[4] - v[0]) * (1 - local_x) * (1 - local_y) +
                        (v[5] - v[1]) * local_x * (1 - local_y) +
                        (v[6] - v[2]) * (1 - local_x) * local_y +
                        (v[7] - v[3]) * local_x * local_y;

            V3f n = make_float3(gx0, gy0, gz0);
            float grad_len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
            if (grad_len > 1e-6f) {
                n.x /= grad_len;
                n.y /= grad_len;
                n.z /= grad_len;
            } else {
                // Fallback: use edge direction
                V3f edge_dir = make_float3(off1.x - off0.x, off1.y - off0.y, off1.z - off0.z);
                float edge_len = sqrtf(edge_dir.x*edge_dir.x + edge_dir.y*edge_dir.y + edge_dir.z*edge_dir.z);
                if (edge_len > 1e-6f) {
                    n = make_float3(edge_dir.x / edge_len, edge_dir.y / edge_len, edge_dir.z / edge_len);
                } else {
                    n = make_float3(0.0f, 0.0f, 1.0f);
                }
            }

            // planar equation (ax + by + cz - d = 0)
            float d_plane = n.x * p.x + n.y * p.y + n.z * p.z;

            // Quadric Matrix
            Quadric q_plane;
            q_plane.a2 = n.x * n.x; 
            q_plane.ab = n.x * n.y; 
            q_plane.ac = n.x * n.z; 
            q_plane.ad = - d_plane * n.x;

            q_plane.b2 = n.y * n.y; 
            q_plane.bc = n.y * n.z; 
            q_plane.bd = - d_plane * n.y;

            q_plane.c2 = n.z * n.z; 
            q_plane.cd = -d_plane * n.z;
            q_plane.d2 = d_plane * d_plane;

            q_accum.add(q_plane);
        }

        // Compute centroid
        if (num_points > 0.0f) {
            centroid.x /= num_points;
            centroid.y /= num_points;
            centroid.z /= num_points;
        }

        int patch_out_idx = outputBaseIdx + (patch_index - firstPatch);
        patchQuadrics[patch_out_idx] = q_accum;
        patchCentroids[patch_out_idx] = centroid;
    }
}

__global__ void solveQemKernel(int totalPatches, const Quadric* patchQuadrics,
                                const V3f* patchCentroids, V3f* verts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalPatches) return;

    Quadric q = patchQuadrics[i];

    // 3x3 matrix
    // | a2 ab ac | |x|   |-ad|
    // | ab b2 bc | |y| = |-bd|
    // | ac bc c2 | |z|   |-cd|
    float det = q.a2 * (q.b2 * q.c2 - q.bc * q.bc) -
                q.ab * (q.ab * q.c2 - q.bc * q.ac) +
                q.ac * (q.ab * q.bc - q.b2 * q.ac);

    if (fabsf(det) < 1e-8f) {
        // Singular/near-singular matrix: use MC centroid as fallback
        verts[i] = patchCentroids[i];
        return;
    }

    float inv_det = 1.0f / det;
    
    // inverted matrix (Cramer's rule)
    float x = inv_det * (
        -q.ad * (q.b2 * q.c2 - q.bc * q.bc) +
         q.bd * (q.ab * q.c2 - q.ac * q.bc) -
         q.cd * (q.ab * q.bc - q.ac * q.b2)
    );
    float y = inv_det * (
         q.ad * (q.ac * q.bc - q.ab * q.c2) -
         q.bd * (q.a2 * q.c2 - q.ac * q.ac) +
         q.cd * (q.ab * q.ac - q.a2 * q.bc)
    );
    float z = inv_det * (
        -q.ad * (q.ab * q.bc - q.ac * q.b2) +
         q.bd * (q.a2 * q.bc - q.ab * q.ac) -
         q.cd * (q.a2 * q.b2 - q.ab * q.ab)
    );

    verts[i] = make_float3(x, y, z);
}

__device__ float scalar_triple_product(const V3f& v1, const V3f& v2,
                                        const V3f& v3, const V3f& ref) {
    V3f v2_ref = {v2.x - ref.x, v2.y - ref.y, v2.z - ref.z};
    V3f v3_ref = {v3.x - ref.x, v3.y - ref.y, v3.z - ref.z};
    V3f cross_product = {
        v2_ref.y * v3_ref.z - v2_ref.z * v3_ref.y,
        v2_ref.z * v3_ref.x - v2_ref.x * v3_ref.z,
        v2_ref.x * v3_ref.y - v2_ref.y * v3_ref.x
    };
    V3f v1_ref = {v1.x - ref.x, v1.y - ref.y, v1.z - ref.z};
    return v1_ref.x * cross_product.x + v1_ref.y * cross_product.y + v1_ref.z * cross_product.z;
}


// Helper struct to pack 3D coordinates into a single 64-bit integer key
struct CoordKey {
    static __host__ __device__ int64_t get_key(int x, int y, int z) {
        return (static_cast<int64_t>(x) << 40) | (static_cast<int64_t>(y) << 20) | static_cast<int64_t>(z);
    }

    static __host__ __device__ thrust::tuple<int, int, int> get_coords(int64_t key) {
        int x = (key >> 40) & 0xFFFFF;
        int y = (key >> 20) & 0xFFFFF;
        int z = key & 0xFFFFF;
        return thrust::make_tuple(x, y, z);
    }
};

// Build a hash map: coord -> voxel index
__global__ void buildCoordMap(const int* coords, int N, int64_t* coord_keys, int* voxel_indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int x = coords[i*3+0];
    int y = coords[i*3+1];
    int z = coords[i*3+2];
    coord_keys[i] = CoordKey::get_key(x, y, z);
    voxel_indices[i] = i;
}

// Lookup voxel index by coordinates
__device__ int findVoxelIndex(const int64_t* sorted_keys, const int* sorted_indices,
                              int N, int x, int y, int z) {
    int64_t query_key = CoordKey::get_key(x, y, z);

    // Binary search
    const int64_t* first = sorted_keys;
    int count = N;
    while (count > 0) {
        int step = count / 2;
        const int64_t* mid = first + step;
        if (*mid < query_key) {
            first = mid + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    int found_idx = first - sorted_keys;

    if (found_idx < N && sorted_keys[found_idx] == query_key) {
        return sorted_indices[found_idx];
    }
    return -1;
}

// Count MC vertices (entering/exiting edges)
__global__ void countMcVertsKernel(const int* coords, const float* corners, int N, float iso,
                                    int* mcVertCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = corners[i*8 + j];
    }

    // Count entering/exiting edges (X, Y, Z directions)
    int count = 0;
    float d0 = v[0]; // corner 000
    float dx = v[1]; // corner 100
    float dy = v[2]; // corner 010
    float dz = v[4]; // corner 001

    if ((d0 < iso && dx >= iso) || (dx < iso && d0 >= iso)) count++;
    if ((d0 < iso && dy >= iso) || (dy < iso && d0 >= iso)) count++;
    if ((d0 < iso && dz >= iso) || (dz < iso && d0 >= iso)) count++;

    mcVertCount[i] = count;
}

// Index MC vertices and record their types
__global__ void indexMcVertsKernel(const int* coords, const float* corners, int N, float iso,
                                    const int* mcVertPrefix,
                                    int* mcVertToVoxel, uint8_t* mcVertType,
                                    float* mcVertEdgeP, float* mcVertEdgeN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int vx = coords[i*3+0];
    int vy = coords[i*3+1];
    int vz = coords[i*3+2];

    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = corners[i*8 + j];
    }

    int outputIdx = mcVertPrefix[i];
    float d0 = v[0];
    float vals[3] = {v[1], v[2], v[4]}; // dx, dy, dz

    V3f v0 = make_float3(vx, vy, vz);
    V3f vn[3] = {
        make_float3(vx+1, vy, vz),
        make_float3(vx, vy+1, vz),
        make_float3(vx, vy, vz+1)
    };

    for (int dim = 0; dim < 3; dim++) {
        float d = vals[dim];
        bool entering = d0 < iso && d >= iso;
        bool exiting = d < iso && d0 >= iso;

        if (entering || exiting) {
            mcVertToVoxel[outputIdx] = i;
            mcVertType[outputIdx] = (exiting ? 3 : 0) + dim;

            // Store edge endpoints for concavity testing later
            if (entering) {
                mcVertEdgeP[outputIdx*3+0] = vn[dim].x;
                mcVertEdgeP[outputIdx*3+1] = vn[dim].y;
                mcVertEdgeP[outputIdx*3+2] = vn[dim].z;
                mcVertEdgeN[outputIdx*3+0] = v0.x;
                mcVertEdgeN[outputIdx*3+1] = v0.y;
                mcVertEdgeN[outputIdx*3+2] = v0.z;
            } else {
                mcVertEdgeP[outputIdx*3+0] = v0.x;
                mcVertEdgeP[outputIdx*3+1] = v0.y;
                mcVertEdgeP[outputIdx*3+2] = v0.z;
                mcVertEdgeN[outputIdx*3+0] = vn[dim].x;
                mcVertEdgeN[outputIdx*3+1] = vn[dim].y;
                mcVertEdgeN[outputIdx*3+2] = vn[dim].z;
            }
            outputIdx++;
        }
    }
}

__device__ uint8_t get_neighbor_cell_code(
    int nx, int ny, int nz,
    const int64_t* sorted_coord_keys, const int* sorted_voxel_indices, int totalVoxels,
    const float* all_corners, float iso)
{
    int neighbor_idx = findVoxelIndex(sorted_coord_keys, sorted_voxel_indices, totalVoxels, nx, ny, nz);
    if (neighbor_idx == -1) {
        return 0; // Neighbor doesn't exist, cannot resolve, return empty code
    }

    // If neighbor exists, compute its code from its 8 corner values
    int code = 0;
    const float* neighbor_corners = all_corners + neighbor_idx * 8;
    if (neighbor_corners[0] >= iso) code |= 1;
    if (neighbor_corners[1] >= iso) code |= 2;
    if (neighbor_corners[2] >= iso) code |= 4;
    if (neighbor_corners[3] >= iso) code |= 8;
    if (neighbor_corners[4] >= iso) code |= 16;
    if (neighbor_corners[5] >= iso) code |= 32;
    if (neighbor_corners[6] >= iso) code |= 64;
    if (neighbor_corners[7] >= iso) code |= 128;
    return code;
}

// Count patches per voxel
__global__ void countPatchesKernel(const int* coords, const float* corners, int N, float iso,
                                    uint8_t* cellCodes, int* patchCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = corners[i*8 + j];
    }

    // Compute cube code
    int cubeCode = 0;
    if (v[0] >= iso) cubeCode |= 1;
    if (v[1] >= iso) cubeCode |= 2;
    if (v[2] >= iso) cubeCode |= 4;
    if (v[3] >= iso) cubeCode |= 8;
    if (v[4] >= iso) cubeCode |= 16;
    if (v[5] >= iso) cubeCode |= 32;
    if (v[6] >= iso) cubeCode |= 64;
    if (v[7] >= iso) cubeCode |= 128;

    cellCodes[i] = cubeCode;
    int numPatches = mcFirstPatchIndex[cubeCode + 1] - mcFirstPatchIndex[cubeCode];
    patchCount[i] = numPatches;
}

// Create dual vertices (patch centers)
__global__ void createDualVertsKernel(const int* coords, const float* corners, int N, float iso,
                                       const uint8_t* cellCodes, const int* patchPrefix,
                                       V3f* verts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint8_t cubeCode = cellCodes[i];
    if (cubeCode == 0 || cubeCode == 255) return;

    int vx = coords[i*3+0];
    int vy = coords[i*3+1];
    int vz = coords[i*3+2];

    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = corners[i*8 + j];
    }

    int firstPatch = mcFirstPatchIndex[cubeCode];
    int lastPatch = mcFirstPatchIndex[cubeCode + 1];
    int outputIdx = patchPrefix[i];

    for (int patch_index = firstPatch; patch_index < lastPatch; ++patch_index) {
        V3f p = make_float3(0, 0, 0);
        float num = 0;

        int firstEdge = mcFirstEdgeIndex[patch_index];
        int lastEdge = mcFirstEdgeIndex[patch_index + 1];

        for (int edge_index = firstEdge; edge_index < lastEdge; ++edge_index) {
            int eid = mcEdgeIndex[edge_index];
            int ex = dmcEdgeLocations[eid][0];
            int ey = dmcEdgeLocations[eid][1];
            int ez = dmcEdgeLocations[eid][2];
            int dim = dmcEdgeLocations[eid][3];

            // Compute MC vertex on this edge
            int c0 = (ex) + (ey)*2 + (ez)*4;
            int c1_offset[3] = {1, 2, 4};
            int c1 = c0 + c1_offset[dim];

            float v0_val = v[c0];
            float v1_val = v[c1];
            float t = (iso - v0_val) / (v1_val - v0_val + 1e-30f);
            t = fminf(fmaxf(t, 0.f), 1.f);

            int3 off0 = dmcCornerOffset[c0];
            int3 off1 = dmcCornerOffset[c1];

            float px = vx + off0.x + t * (off1.x - off0.x);
            float py = vy + off0.y + t * (off1.y - off0.y);
            float pz = vz + off0.z + t * (off1.z - off0.z);

            p.x += px;
            p.y += py;
            p.z += pz;
            num += 1.0f;
        }

        p.x /= num;
        p.y /= num;
        p.z /= num;

        verts[outputIdx++] = p;
    }
}

// Get patch index for a given voxel and edge
__device__ int getPatchIndex(int voxel_idx, int edge_id,
                              const uint8_t* cellCodes, const int* patchPrefix) {
    uint8_t cellCode = cellCodes[voxel_idx];
    int8_t offset = dmcEdgeOffset[cellCode][edge_id];
    if (offset < 0) return -1;
    return patchPrefix[voxel_idx] + offset;
}

// Create quads
__global__ void createQuadsKernel(const int* coords, int N,
                                   const uint8_t* cellCodes, const int* patchPrefix,
                                   const int* mcVertPrefix, const int* mcVertToVoxel,
                                   const uint8_t* mcVertType,
                                   const int64_t* sorted_coord_keys, const int* sorted_voxel_indices,
                                   int totalVoxels, int totalMcVerts,
                                   Quad* quads) {
    int quad_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (quad_index >= mcVertPrefix[N]) return; // Total MC verts = quads
    if (quad_index >= totalMcVerts) return;

    int voxel_idx = mcVertToVoxel[quad_index];
    uint8_t vert_type = mcVertType[quad_index];

    int vx = coords[voxel_idx*3+0];
    int vy = coords[voxel_idx*3+1];
    int vz = coords[voxel_idx*3+2];

    // Build quad from 4 neighboring cells
    Quad q = {-1, -1, -1, -1};

    for (int i = 0; i < 4; ++i) {
        int dx = dmcQuad[vert_type][i][0];
        int dy = dmcQuad[vert_type][i][1];
        int dz = dmcQuad[vert_type][i][2];
        int eid = dmcQuad[vert_type][i][3];

        int nx = vx + dx;
        int ny = vy + dy;
        int nz = vz + dz;

        // Find neighbor voxel
        int neighbor_idx = findVoxelIndex(sorted_coord_keys, sorted_voxel_indices,
                                          totalVoxels, nx, ny, nz);

        if (neighbor_idx >= 0) {
            int patch_idx = getPatchIndex(neighbor_idx, eid, cellCodes, patchPrefix);
            if (i == 0) q.v0 = patch_idx;
            else if (i == 1) q.v1 = patch_idx;
            else if (i == 2) q.v2 = patch_idx;
            else if (i == 3) q.v3 = patch_idx;
        }
    }

    if (q.v0 < 0 || q.v1 < 0 || q.v2 < 0 || q.v3 < 0) {
        q.v0 = q.v1 = q.v2 = q.v3 = -1;
    }

    quads[quad_index] = q;
}

// Gather corners kernel for from_points
__global__ void gather_corners_kernel_dual(
    const int* __restrict__ active_voxel_coords, int N,
    const int64_t* __restrict__ sorted_point_keys,
    const float* __restrict__ sorted_point_values, int M,
    float default_value, float* __restrict__ out_corners) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int vx = active_voxel_coords[i * 3 + 0];
    int vy = active_voxel_coords[i * 3 + 1];
    int vz = active_voxel_coords[i * 3 + 2];

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int3 offset = dmcCornerOffset[j];
        int cx = vx + offset.x;
        int cy = vy + offset.y;
        int cz = vz + offset.z;

        int64_t query_key = CoordKey::get_key(cx, cy, cz);

        const int64_t* first = sorted_point_keys;
        int count = M;
        while (count > 0) {
            int step = count / 2;
            const int64_t* mid = first + step;
            if (*mid < query_key) {
                first = mid + 1;
                count -= step + 1;
            } else {
                count = step;
            }
        }
        int found_idx = first - sorted_point_keys;

        float value = default_value;
        if (found_idx < M && sorted_point_keys[found_idx] == query_key) {
            value = sorted_point_values[found_idx];
        }
        out_corners[i * 8 + j] = value;
    }
}

struct is_valid_quad {
    __host__ __device__
    bool operator()(const Quad& q) const {
        return q.v0 != -1; 
    }
};

// Main sparse dual marching cubes function
std::tuple<thrust::device_vector<V3f>, thrust::device_vector<Quad>>
_sparse_dual_marching_cubes(const int* d_coords, const float* d_corners, int N, float iso, cudaStream_t stream) {

    const int threads = 256;
    int blocks = (N + threads - 1) / threads;

    thrust::device_vector<V3f> vertices;
    thrust::device_vector<Quad> quads;

    if (N == 0) return {std::move(vertices), std::move(quads)};

    // Step 1: Build coordinate lookup map
    thrust::device_vector<int64_t> coord_keys(N);
    thrust::device_vector<int> voxel_indices(N);

    buildCoordMap<<<blocks, threads, 0, stream>>>(
        d_coords, N,
        thrust::raw_pointer_cast(coord_keys.data()),
        thrust::raw_pointer_cast(voxel_indices.data())
    );

    // Sort by coordinate keys
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(coord_keys.begin(), voxel_indices.begin()));
    thrust::sort(thrust::cuda::par.on(stream), zipped, zipped + N);

    // Step 2: Count MC vertices
    thrust::device_vector<int> mcVertCount(N);
    countMcVertsKernel<<<blocks, threads, 0, stream>>>(
        d_coords, d_corners, N, iso,
        thrust::raw_pointer_cast(mcVertCount.data())
    );

    thrust::device_vector<int> mcVertPrefix(N + 1);
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                        mcVertCount.begin(), mcVertCount.end(),
                        mcVertPrefix.begin());

    thrust::transform(thrust::cuda::par.on(stream),
                  mcVertPrefix.begin() + (N - 1), 
                  mcVertPrefix.begin() + N,
                  mcVertCount.begin() + (N - 1),  
                  mcVertPrefix.begin() + N,      
                  thrust::plus<int>());

    int totalMcVerts = 0;
    cudaMemcpyAsync(&totalMcVerts, thrust::raw_pointer_cast(mcVertPrefix.data()) + N,
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (totalMcVerts == 0) return {std::move(vertices), std::move(quads)};

    // Step 3: Index MC vertices
    thrust::device_vector<int> mcVertToVoxel(totalMcVerts);
    thrust::device_vector<uint8_t> mcVertType(totalMcVerts);
    thrust::device_vector<float> mcVertEdgeP(totalMcVerts * 3);
    thrust::device_vector<float> mcVertEdgeN(totalMcVerts * 3);

    indexMcVertsKernel<<<blocks, threads, 0, stream>>>(
        d_coords, d_corners, N, iso,
        thrust::raw_pointer_cast(mcVertPrefix.data()),
        thrust::raw_pointer_cast(mcVertToVoxel.data()),
        thrust::raw_pointer_cast(mcVertType.data()),
        thrust::raw_pointer_cast(mcVertEdgeP.data()),
        thrust::raw_pointer_cast(mcVertEdgeN.data())
    );

    // Step 4: Count patches
    thrust::device_vector<uint8_t> cellCodes(N);
    thrust::device_vector<int> patchCount(N);

    countPatchesKernel<<<blocks, threads, 0, stream>>>(
        d_coords, d_corners, N, iso,
        thrust::raw_pointer_cast(cellCodes.data()),
        thrust::raw_pointer_cast(patchCount.data())
    );

    thrust::device_vector<int> patchPrefix(N + 1);
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                        patchCount.begin(), patchCount.end(),
                        patchPrefix.begin());

    thrust::transform(thrust::cuda::par.on(stream),
                  patchPrefix.begin() + (N - 1),
                  patchPrefix.begin() + N,
                  patchCount.begin() + (N - 1),
                  patchPrefix.begin() + N,
                  thrust::plus<int>());

    int totalPatches = 0;
    cudaMemcpyAsync(&totalPatches, thrust::raw_pointer_cast(patchPrefix.data()) + N,
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (totalPatches == 0) return {std::move(vertices), std::move(quads)};

    // Step 5: Create dual vertices w/o QEM
    vertices.resize(totalPatches);
    createDualVertsKernel<<<blocks, threads, 0, stream>>>(
        d_coords, d_corners, N, iso,
        thrust::raw_pointer_cast(cellCodes.data()),
        thrust::raw_pointer_cast(patchPrefix.data()),
        thrust::raw_pointer_cast(vertices.data())
    );

    // // Step 5: Create dual vertices w/ QEM
    // thrust::device_vector<Quadric> patchQuadrics(totalPatches);
    // thrust::device_vector<V3f> patchCentroids(totalPatches);
    // calculatePatchQuadricsKernel<<<blocks, threads, 0, stream>>>(
    //     d_coords, d_corners, N, iso,
    //     thrust::raw_pointer_cast(cellCodes.data()),
    //     thrust::raw_pointer_cast(patchPrefix.data()),
    //     thrust::raw_pointer_cast(patchQuadrics.data()),
    //     thrust::raw_pointer_cast(patchCentroids.data())
    // );

    // // Step 5.5: solve QEM
    // vertices.resize(totalPatches);
    // int patch_blocks = (totalPatches + threads - 1) / threads;
    // solveQemKernel<<<patch_blocks, threads, 0, stream>>>(
    //     totalPatches,
    //     thrust::raw_pointer_cast(patchQuadrics.data()),
    //     thrust::raw_pointer_cast(patchCentroids.data()),
    //     thrust::raw_pointer_cast(vertices.data())
    // );

    // Step 6: Create quads
    quads.resize(totalMcVerts);
    int quad_blocks = (totalMcVerts + threads - 1) / threads;

    createQuadsKernel<<<quad_blocks, threads, 0, stream>>>(
        d_coords, N,
        thrust::raw_pointer_cast(cellCodes.data()),
        thrust::raw_pointer_cast(patchPrefix.data()),
        thrust::raw_pointer_cast(mcVertPrefix.data()),
        thrust::raw_pointer_cast(mcVertToVoxel.data()),
        thrust::raw_pointer_cast(mcVertType.data()),
        thrust::raw_pointer_cast(coord_keys.data()),
        thrust::raw_pointer_cast(voxel_indices.data()),
        N,
        totalMcVerts,
        thrust::raw_pointer_cast(quads.data())
    );

    cudaStreamSynchronize(stream);

    int num_valid_quads = thrust::count_if(thrust::cuda::par.on(stream),
                                       quads.begin(), quads.end(),
                                       is_valid_quad());

    if (num_valid_quads == 0) {
        vertices.clear();
        quads.clear();
        return {std::move(vertices), std::move(quads)};
    }

    thrust::device_vector<Quad> valid_quads(num_valid_quads);

    thrust::copy_if(thrust::cuda::par.on(stream),
                    quads.begin(), quads.end(),
                    valid_quads.begin(),
                    is_valid_quad());

    cudaStreamSynchronize(stream);


    return {std::move(vertices), std::move(valid_quads)};
}

// From points wrapper
std::tuple<thrust::device_vector<V3f>, thrust::device_vector<Quad>>
_sparse_dual_marching_cubes_from_points(const int* d_coords, const float* d_point_values,
                                        int N, float iso, float default_value, cudaStream_t stream) {
    const int threads = 256;

    if (N == 0) {
        thrust::device_vector<V3f> vertices;
        thrust::device_vector<Quad> quads;
        return {std::move(vertices), std::move(quads)};
    }

    // Step 1: Create sorted point keys for efficient lookup
    thrust::device_vector<int64_t> point_keys(N);
    thrust::device_vector<float> point_values(N);

    thrust::transform(thrust::cuda::par.on(stream),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(N),
                     point_keys.begin(),
                     [d_coords] __device__ (int i) {
                         int x = d_coords[i*3+0];
                         int y = d_coords[i*3+1];
                         int z = d_coords[i*3+2];
                         return CoordKey::get_key(x, y, z);
                     });

    cudaMemcpyAsync(thrust::raw_pointer_cast(point_values.data()),
                    d_point_values, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(point_keys.begin(), point_values.begin()));
    thrust::sort(thrust::cuda::par.on(stream), zipped, zipped + N);

    // Step 2: Gather corners for each voxel
    thrust::device_vector<float> corners(N * 8);

    int blocks = (N + threads - 1) / threads;
    gather_corners_kernel_dual<<<blocks, threads, 0, stream>>>(
        d_coords, N,
        thrust::raw_pointer_cast(point_keys.data()),
        thrust::raw_pointer_cast(point_values.data()),
        N, default_value,
        thrust::raw_pointer_cast(corners.data())
    );

    // Step 3: Call sparse dual marching cubes
    return _sparse_dual_marching_cubes(
        d_coords,
        thrust::raw_pointer_cast(corners.data()),
        N, iso, stream
    );
}

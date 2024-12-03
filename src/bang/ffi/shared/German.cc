/*
 * German.cpp
 *
 *  Created on: Sep 21, 2015
 *      Author: qixia.yuan
 */

#include "German.h"
#include <float.h>
#include <functional>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

German::German() {
  // TODO Auto-generated constructor stub
}

German::~German() {
  // TODO Auto-generated destructor stub
}
/**
 * n is the number of nodes;
 * size is the length of each trajectory;
 * m is the number of trajectories;
 */
float German::computePstr(std::vector<bool> array, int size, int n, int m) {
  int prefix = m * n;
  int german_n = size / 2;
  int startIndex;

  float mean[m];
  float meansquare[m];
  float sj[m];
  float values[german_n];
  float grandmean;
  float within, between;
  float variance;
  float psrf; // potential scale reduction factor
  float varianceW, varianceB, varianceV;
  float covwxsquare, covwx, covwb;
  float mean_meansquare = 0;
  float df;

  within = 0;
  grandmean = 0;
  mean_meansquare = 0;

  clock_t begin, end, begin1, end1;
  double time_spent;
  double time_spent1 = 0;
  begin = clock();

  for (int i = 0; i < m; i++) {
    // trajectories.get(i).clear(0, twon-n);//discard first n steps;
    mean[i] = 0;
    begin1 = clock();
    for (int j = 0; j < german_n; j++) {
      startIndex = prefix * (j + german_n) + i * n;
      // printf("startIndex=%d\n",startIndex);
      values[j] = getFloatValue(array, startIndex, startIndex + n - 1);
      // printf("value[%d]=%f\n",j,values[j]);
      mean[i] += values[j];
    }
    end1 = clock();
    time_spent1 = time_spent1 + (double)(end1 - begin1) / CLOCKS_PER_SEC;
    printf(" *** CPU german_n loop execution time: %f s*** \n", time_spent1);
    mean[i] = mean[i] / n;
    meansquare[i] = (float)pow(mean[i], 2);
    mean_meansquare += meansquare[i];
    // mean[i]=(double)(trajectories.get(i).cardinality())/n;
    grandmean += mean[i];
    sj[i] = 0;
    for (int j = 0; j < german_n; j++) {
      // sj[i]+=Math.pow((Integer)(trajectories.get(i).get(j))-mean[i],2);
      sj[i] += (float)pow(values[j] - mean[i], 2);
    }

    sj[i] =
        sj[i] / (german_n - 1); // the variance, note here we use 1/(german_n-1)
    within += sj[i];
  }

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf(" *** CPU within execution time: %f s*** \n", time_spent);

  within = within / m;

  //********************************
  grandmean = grandmean / m;
  mean_meansquare = mean_meansquare / m;
  between = 0;
  for (int i = 0; i < m; i++) {
    between += (float)pow((grandmean - mean[i]), 2);
  }
  between = between * (german_n) / (m - 1);
  // between = n * StatUtils.variance(mean);
  varianceW = 0;
  for (int i = 0; i < m; i++) {
    varianceW += pow(sj[i] - within, 2);
  }
  varianceW = varianceW / (m - 1);

  // varianceW = StatUtils.variance(sj);
  varianceW = varianceW / m;

  varianceB = (2 * between * between) / (m - 1);
  covwxsquare = 0;
  covwx = 0;
  for (int i = 0; i < m; i++) {
    covwxsquare += (sj[i] - within) * (mean[i] * mean[i] - mean_meansquare);
    covwx += (sj[i] - within) * (mean[i] - grandmean);
  }
  covwxsquare = covwxsquare / (m - 1);
  covwx = covwx / (m - 1);

  // covwxsquare = covariance.covariance(sj, meansquare);
  // covwx = covariance.covariance(sj, mean);

  covwb = german_n / m * (covwxsquare - 2 * grandmean * covwx);
  varianceV =
      (pow(german_n - 1, 2) * varianceW + pow(1 + 1 / m, 2) * varianceB +
       2 * (1 / m + 1) * (german_n - 1) * covwb) /
      pow(german_n, 2);

  variance = (1 - 1 / (float)german_n) * within +
             ((m + 1) / (float)(m * german_n)) * between;
  // System.out.println("variance:"+variance);
  if (varianceV == 0)
    return FLT_MAX;

  df = (2 * variance * variance) / varianceV;
  psrf = sqrt(variance / within * (df + 3) / (df + 1));
  return psrf;
}
/**
 * n is the number of nodes;
 * size is the length of each trajectory;
 * m is the number of trajectories;
 */
float German::computePstr(float *stat, int size, int n, int m) {
  // int prefix = m * n;
  int german_n = size / 2;
  // int startIndex;

  float mean[m];
  float meansquare[m];
  float sj[m];
  float grandmean;
  float within, between;
  float variance;
  float psrf; // potential scale reduction factor
  float varianceW, varianceB, varianceV;
  float covwxsquare, covwx, covwb;
  float mean_meansquare = 0;
  float df;

  within = 0;
  grandmean = 0;
  mean_meansquare = 0;

  // clock_t begin, end,begin1,end1;
  // double time_spent;
  // double time_spent1=0;
  // begin = clock();

  for (int i = 0; i < m; i++) {
    mean[i] = stat[i * 2];
    sj[i] = stat[i * 2 + 1];
    // trajectories.get(i).clear(0, twon-n);//discard first n steps;
    meansquare[i] = (float)pow(mean[i], 2);
    mean_meansquare += meansquare[i];
    // mean[i]=(double)(trajectories.get(i).cardinality())/n;
    grandmean += mean[i];
    within += sj[i];
  }

  /*		end = clock();
                          time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
                          printf(" *** CPU within execution time: %f s*** \n",
     time_spent);*/

  within = within / m;

  //********************************
  grandmean = grandmean / m;
  mean_meansquare = mean_meansquare / m;
  between = 0;
  for (int i = 0; i < m; i++) {
    between += (float)pow((grandmean - mean[i]), 2);
  }
  between = between * (german_n) / (m - 1);
  // between = n * StatUtils.variance(mean);
  varianceW = 0;
  for (int i = 0; i < m; i++) {
    varianceW += pow(sj[i] - within, 2);
  }
  varianceW = varianceW / (m - 1);

  // varianceW = StatUtils.variance(sj);
  varianceW = varianceW / m;

  varianceB = (2 * between * between) / (m - 1);
  covwxsquare = 0;
  covwx = 0;
  for (int i = 0; i < m; i++) {
    covwxsquare += (sj[i] - within) * (mean[i] * mean[i] - mean_meansquare);
    covwx += (sj[i] - within) * (mean[i] - grandmean);
  }
  covwxsquare = covwxsquare / (m - 1);
  covwx = covwx / (m - 1);

  // covwxsquare = covariance.covariance(sj, meansquare);
  // covwx = covariance.covariance(sj, mean);

  covwb = german_n / m * (covwxsquare - 2 * grandmean * covwx);
  varianceV =
      (pow(german_n - 1, 2) * varianceW + pow(1 + 1 / m, 2) * varianceB +
       2 * (1 / m + 1) * (german_n - 1) * covwb) /
      pow(german_n, 2);

  variance = (1 - 1 / (float)german_n) * within +
             ((m + 1) / (float)(m * german_n)) * between;
  // System.out.println("variance:"+variance);
  if (varianceV == 0)
    return FLT_MAX;

  df = (2 * variance * variance) / varianceV;
  psrf = sqrt(variance / within * (df + 3) / (df + 1));
  return psrf;
}
float German::getFloatValue(std::vector<bool> array, int startIndex,
                            int endIndex) {
  std::vector<bool> newarray(endIndex - startIndex + 1);
  for (int i = 0; i < endIndex - startIndex + 1; i++) {
    newarray[i] = true; // array[startIndex+i];
  }
  // newarray.assign(array.begin()+startIndex,array.begin()+endIndex);
  std::hash<std::vector<bool>> h;
  double v = h(newarray);
  std::cout << h(newarray);
  printf("startIndex=%d,endIndex=%d,v=%f, ", startIndex, endIndex, v);
  for (int i = 0; i < endIndex - startIndex + 1; i++) {
    std::cout << "newarray[" << i << "]=" << newarray[i];
    // printf("newarray[%d]=%d,array=%d ",i,newarray[i],array[startIndex+i]);
  }
  printf("\n");
  return v;
  /*float result = 0;
  float time = 1;
  //printf("array adress %d",&array);
  while (endIndex - startIndex >= 32) {
          result += time
                          * (float) BoolArrayToInt(array, startIndex, startIndex
  + 31); startIndex += 32; time *= (float) pow(2, 32);
  }
  result += time * (float) BoolArrayToInt(array, startIndex, endIndex);
  return result;*/
}
unsigned int German::BoolArrayToInt(std::vector<bool> array, int startIndex,
                                    int endIndex) {
  int size = endIndex - startIndex + 1;
  if (endIndex - startIndex >= 32)
    throw "Can only fit 32 bits in a uint";
  // printf("array adress %d",&array);
  unsigned int r = 0;
  for (int i = 0; i < size; i++) {
    if (array[startIndex + i])
      r |= 1 << (size - i);
    // printf("array[%d]=%d\n",startIndex+i,array[startIndex+i]);
  }

  return r;
}

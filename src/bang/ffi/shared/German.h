/*
 * German.h
 *
 *  Created on: Sep 21, 2015
 *      Author: qixia.yuan
 */

#ifndef GERMAN_H_
#define GERMAN_H_
#include <vector>

class German {
public:
	German();
	virtual ~German();
	float computePstr(std::vector<bool> array, int size, int n, int m);
	float computePstr(float* trajectory, int size, int n, int m);
	float getFloatValue(std::vector<bool> array,int startIndex,int endIndex);
	unsigned int BoolArrayToInt(std::vector<bool> array,int startIndex,int endIndex);
};

#endif /* GERMAN_H_ */

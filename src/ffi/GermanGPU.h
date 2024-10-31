/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class simulationMethod_GermanGPU */

#ifndef _Included_simulationMethod_GermanGPU
#define _Included_simulationMethod_GermanGPU
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     simulationMethod_GermanGPU
 * Method:    initialGPUPBN
 * Signature: (LPBN/PBN;)V
 */
JNIEXPORT void JNICALL Java_simulationMethod_GermanGPU_initialGPUPBN
  (JNIEnv *, jclass, jobject);

/*
 * Class:     simulationMethod_GermanGPU
 * Method:    initialGPUExpression
 * Signature: ([J[J)V
 */
JNIEXPORT void JNICALL Java_simulationMethod_GermanGPU_initialGPUExpression
  (JNIEnv *, jclass, jlongArray, jlongArray);

/*
 * Class:     simulationMethod_GermanGPU
 * Method:    initialGerman
 * Signature: ([D)V
 */
JNIEXPORT void JNICALL Java_simulationMethod_GermanGPU_initialGerman
  (JNIEnv *, jclass, jdoubleArray);

/*
 * Class:     simulationMethod_GermanGPU
 * Method:    setOutputName
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_simulationMethod_GermanGPU_setOutputName
  (JNIEnv *, jclass, jstring);

/*
 * Class:     simulationMethod_GermanGPU
 * Method:    run
 * Signature: ()[D
 */
JNIEXPORT jdoubleArray JNICALL Java_simulationMethod_GermanGPU_run
  (JNIEnv *, jclass);

/*
 * Class:     simulationMethod_GermanGPU
 * Method:    getCudaDeviceCount
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_simulationMethod_GermanGPU_getCudaDeviceCount
  (JNIEnv *, jclass);

#ifdef __cplusplus
}
#endif
#endif

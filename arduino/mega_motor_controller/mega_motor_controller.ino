/**
 *  Name : mega_stepmotor_controller
 *
 *  Author : Tony Guo
 *  
 *  Country : Taiwan
 *
 *  Date : 5 Oct, 2019 
 */


#include "serial_read.h"

#define X_STEP_PIN         54
#define X_DIR_PIN          55
#define X_ENABLE_PIN       38
#define X_MIN_PIN           3
#define X_MAX_PIN           2

#define Y_STEP_PIN         60
#define Y_DIR_PIN          61
#define Y_ENABLE_PIN       56
#define Y_MIN_PIN          14
#define Y_MAX_PIN          15

#define Z_STEP_PIN         46
#define Z_DIR_PIN          48
#define Z_ENABLE_PIN       62
#define Z_MIN_PIN          18
#define Z_MAX_PIN          19

#define E_STEP_PIN         26
#define E_DIR_PIN          28
#define E_ENABLE_PIN       24


float serialRead_L;
float serialRead_R;


#define StepPerRev 3200 // (rev/step)
bool temp = false;
bool Ttemp = false;


void Run(int n,int _n,int moto,int dir,int spe);

void setup() {

  //serialReadInit();


  pinMode(X_STEP_PIN,OUTPUT);
  pinMode(X_DIR_PIN,OUTPUT);
  pinMode(X_ENABLE_PIN,OUTPUT);
  pinMode(Y_STEP_PIN,OUTPUT);
  pinMode(Y_DIR_PIN,OUTPUT);
  pinMode(Y_ENABLE_PIN,OUTPUT);

  digitalWrite(X_ENABLE_PIN, LOW);
  digitalWrite(Y_ENABLE_PIN, LOW);
  
  while (!Serial); 
}

void loop() {


  //serialRead_L = (float)serialRead(0)/100.0; //0:left
  //serialRead_R = (float)serialRead(1)/100.0; //1:right
  serialRead_L = 0.5;
  serialRead_R = 0.5;


  Run(X_STEP_PIN,X_DIR_PIN,((int)serialRead_L>0)?(true):(false),abs(serialRead_L));
  //Run(Y_STEP_PIN,Y_DIR_PIN,((int)serialRead_R>0)?(true):(false),abs(serialRead_R));
}


void Run(int s,int d,int dir,float spe){ //rad
  digitalWrite(d,(dir)?(HIGH):(LOW));

  int scape = 1000000/(spe*StepPerRev);
  int Tscape = scape*2;

  temp = (micros()%Tscape >scape)?(true):(false);

  if(temp != Ttemp){
    digitalWrite(s,HIGH);
    delayMicroseconds(1);
    digitalWrite(s,LOW);
    Ttemp = temp;
  }

}

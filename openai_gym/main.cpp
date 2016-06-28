/*
 * main.cpp
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#include "SyncTcpServer.h"

int main()
{
  SyncTcpServer syncServer(2345);
  syncServer.server();

  return 0;
}


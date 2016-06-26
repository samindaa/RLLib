/*
 * main.cpp
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#include "SyncTcpServer.h"
#include "RLLibOpenAiGymProxy.h"

class TestSyncTcpServer: public SyncTcpServer
{
  public:
    TestSyncTcpServer() :
        SyncTcpServer(2345)
    {
    }

    virtual ~TestSyncTcpServer()
    {
    }

    std::string toRLLib(const std::string& str)
    {
      std::cout << "ecoh:" << str << std::endl;
      return str;
    }

};

int main()
{
  //TestSyncTcpServer syncServer;
  //syncServer.server();

  RLLibOpenAiGymProxy syncServer;
  syncServer.server();

  return 0;
}


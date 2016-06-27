/*
 * SyncTcpServer.h
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_SYNCTCPSERVER_H_
#define OPENAI_GYM_SYNCTCPSERVER_H_

#include <string>
#include <iostream>
#include <boost/asio.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/thread/thread.hpp>
//
#include "RLLibOpenAiGymProxy.h"

class SyncTcpServer
{
  private:
    typedef boost::shared_ptr<boost::asio::ip::tcp::socket> socket_ptr;
    typedef boost::shared_ptr<RLLibOpenAiGymProxy> proxy_ptr;

  protected:
    unsigned short port;
    boost::asio::io_service io_service;

  public:
    SyncTcpServer(const unsigned short& port);
    virtual ~SyncTcpServer();

    void server();
    static void session(socket_ptr socket, proxy_ptr proxy);
};

#endif /* OPENAI_GYM_SYNCTCPSERVER_H_ */

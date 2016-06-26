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

class SyncTcpServer
{
  private:
    typedef boost::shared_ptr<boost::asio::ip::tcp::socket> socket_ptr;

  protected:
    unsigned short port;
    boost::asio::io_service io_service;

    enum META
    {
      MAX_LEN = 1024
    };

    char buffer[MAX_LEN];

  public:
    SyncTcpServer(const unsigned short& port);
    virtual ~SyncTcpServer();

    void server();
    static void session(socket_ptr socket, SyncTcpServer* server);

    virtual std::string toRLLib(const std::string& str) =0;

};

#endif /* OPENAI_GYM_SYNCTCPSERVER_H_ */

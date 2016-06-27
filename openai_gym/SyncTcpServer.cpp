/*
 * SyncTcpServer.cpp
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#include "SyncTcpServer.h"

SyncTcpServer::SyncTcpServer(const unsigned short& port) :
    port(port)
{

}

SyncTcpServer::~SyncTcpServer()
{
}

void SyncTcpServer::server()
{
  std::cout << "SyncTcpServer::server" << std::endl;
  boost::asio::ip::tcp::acceptor a(io_service,
      boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
  for (;;)
  {
    socket_ptr sock(new boost::asio::ip::tcp::socket(io_service));
    a.accept(*sock);
    proxy_ptr proxy(new RLLibOpenAiGymProxy());
    boost::thread t(boost::bind(session, sock, proxy));
  }
}

void SyncTcpServer::session(socket_ptr socket, proxy_ptr proxy)
{
  try
  {
    std::cout << "SyncTcpServer::session" << std::endl;
    char buffer[1024];
    for (;;)
    {
      boost::system::error_code error;
      size_t length = socket->read_some(boost::asio::buffer(buffer), error);
      if (error == boost::asio::error::eof)
        break; // Connection closed cleanly by peer.
      else if (error)
        throw boost::system::system_error(error); // Some other error.

      std::string data = proxy->toRLLib(std::string(buffer, length));

      boost::asio::write(*socket, boost::asio::buffer(data));
    }
  } catch (std::exception& e)
  {
    std::cerr << "Exception in thread: " << e.what() << "\n";
  }
}


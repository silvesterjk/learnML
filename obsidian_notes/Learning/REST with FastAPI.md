* A program asking another program to do something
* GET /post/Hello
	* GET - Method
	* /post/Hello - Endpoint
* There is also: 
	* Body 
	* Headers
	
![[Pasted image 20240524193503.png]]

* REST is Architectural constrains
	* Should have "client" and "servers"
	* Should use "resource"
		* Whatever the API deals with
		* Whatever it gets upon making a call
	* Should be stateless
		* Does not keep any information about the client
		* For every request the client must send all the required details for the API request to complete successfully
	* Should be cacheable
	* Should have a uniform, hypermedia-driven interface (optional / mostly not implemented)
		* If a resource is related to another resource, there should be an actual link in the response which allows the client to "find" the related sources
		* For example: ![[Pasted image 20240524194708.png]]
	* If the backend uses multiple servers, this should be invisible to the client.
	


1515
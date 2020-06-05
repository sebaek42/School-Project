#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include "restart.h"

int main() {
    int fd, bytesread;
    int BUFSIZE = 1024;
    char buf[BUFSIZE];
    char* msg;

    fd = r_open3("./text/score.txt", O_RDONLY, 777);
    if(fd == -1) {
        msg = "File Open Error\n";
        r_write(STDERR_FILENO, msg, strlen(msg));
        return 0;
    }

    bytesread = r_read(fd, buf, BUFSIZE);
    if (bytesread == -1) {
        msg = "File Open Error\n";
        r_write(STDERR_FILENO, msg, strlen(msg));
        return 0;
    }

    close(fd);
    r_write(STDERR_FILENO, buf, strlen(buf));
    return 0;
}
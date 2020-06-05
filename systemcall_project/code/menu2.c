#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include "restart.h"

int main() {
    char* msg;
    char res[3];
    int fd, answer, checker;
    int i = 0;

    msg = "고기판 몇 개로 하시겠습니까? :";
    r_write(STDOUT_FILENO, msg, strlen(msg));
    scanf("%d", &answer);

    char buf[answer];
    for(i = 0; i < answer; i++) buf[i] = '0';

    remove("./text/map.txt");

    fd = r_open3("./text/map.txt", O_CREAT | O_RDWR, 777);
    if(fd == -1) {
        msg = "file open error..";
        r_write(STDERR_FILENO, msg, strlen(msg));
        return 0;
    }
    
    checker = r_write(fd, buf, answer);
    if(checker == -1) {
        msg = "file open error..";
        r_write(STDERR_FILENO, msg, strlen(msg));
        return 0;
    }

    r_close(fd);

    msg = "수정되었습니다\n";
    r_write(STDERR_FILENO, msg, strlen(msg));
    return 0;

}
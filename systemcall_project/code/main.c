#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <string.h>
#include "restart.h"

void printMenu() {

    char* msg = "\n1. 게임 시작\n2. 고기판 수정\n3. 도움말\n4. 점수확인\n5. 종료\n메뉴를 입력하세요 : ";
    r_write(STDERR_FILENO, msg, strlen(msg));
}

void menuHandler(int answer) {
    pid_t childpid;
    int fd, count = 0;
    char buf[256];
    char score[1024];

    childpid = fork();
    // if err
    if(childpid == -1) {
        perror("Failed to fork");
        return;
    }
    // child code
    if(childpid == 0) {
        switch(answer) {
            case 1:
                execlp("./binary/game_play", "./binary/game_play", NULL);
                perror("Child failed to exec ls");
                break;
            case 2:
                execlp("./binary/menu2", "./binary/menu2", NULL);
                perror("Child failed to exec ls");
                break;
            case 3:
                execlp("./binary/menu3", "./binary/menu3", NULL);
                perror("Child failed to exec ls");
                break;
            case 4:
                execlp("./binary/menu4", "./binary/menu4", NULL);
                perror("Child failed to exec ls");
                break;            
        }
    }
    // parent waits till child to finish
    while(wait(NULL) > 0);

}

void clean_up() {

    int fd;
    char buf[5] = {'0','0','0','0','0'};
    
    remove("./text/map.txt");

    fd = r_open3("./text/map.txt", O_CREAT | O_RDWR, 777);
    
    r_write(fd, buf, 5);
    r_close(fd);
    //unlink("./text/fifo");

}

void signal_handle() {
    char* msg = "\nCAUGHT INTERRUPT!\nPROGRAM EXIT!\n";
    r_write(STDOUT_FILENO, msg, strlen(msg));
    clean_up();
    exit(1);
}

int main() {

    atexit(clean_up);                    // Normal termination handler

    struct sigaction act;
    act.sa_handler = signal_handle;
    act.sa_flags = 0;

    if((sigemptyset(&act.sa_mask) == -1)) {
        perror("failed to set signal");
        return -1;
    }
    if(sigaction(SIGINT, &act, NULL) == -1 || sigaction(SIGTSTP, &act, NULL) == -1) {
        perror("failed signal handling..");
        return -1;
    }

    while(1) {

        printMenu();

        int answer;
        char *msg;

        scanf("%d", &answer);

        switch(answer) {
            case 1:
            case 2:
            case 3:
            case 4:
                menuHandler(answer);
                break;
            case 5:
                break;
            default:
                msg = "잘못된 입력입니다. 다시 입력해주세요..\n";
                r_write(STDERR_FILENO, msg, strlen(msg));
        }
        if(answer==5) break;
    }
    
    return 0;
}

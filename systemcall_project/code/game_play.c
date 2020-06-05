#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#define BUFSIZE 1024


int main() {
	printf("고기 굽기 게임\n===========================================\n");
	int fd, fd2;
	char buf[BUFSIZE];
	char *s_totalscore = malloc(10);
	char name[BUFSIZE];
	char msg[256] = "USER : ";
	int bytesread;
	int byteswritten = 0;
	int count = 0;
	int i = 0;
	int j = 0;
	int b;                                                                                                                                                                                       int done = 0;                                                                                                                                                                                int n = 5;//불판에 올려놓을 수 있는 고기 수 총 5개라고 가정
	int randomloc[100];//파일의 길이 읽기 100개까지 가정
	int score = 0;
	int totalscore = 0;
	long timedif;
	struct timespec tpend[100], tpstart[100];
	int childpid;
	int countnum = 0;
	char* bp;//buf가르킬포인터

	fd = open("./text/map.txt", O_RDONLY);
	if (fd == -1) printf("File Open Error\n");
	bytesread = read(fd, buf, BUFSIZE);
	close(fd);

	n = bytesread;
	printf("불판개수: %d개\n",n);
	srand(time(NULL));
	for (i = 0; i < n; i++) {
		randomloc[i] = rand()%n;
		for(j=0;j<i;j++){
			if(randomloc[i]==randomloc[j]) {
				i = i - 1;
				break;
			}
		}
	}
	bp = buf;

	while (bytesread > 0) {
		if(!((byteswritten = write(STDOUT_FILENO, bp, n)) == -1) && (errno == EINTR)) break;
		
	//while (((byteswritten = write(STDOUT_FILENO, bp, n)) == -1) && (errno == EINTR));
		if (byteswritten <= 0)//진짜 에러거나 Eof일때 break
			break;
		bytesread -= byteswritten;
		bp += byteswritten;
	}

	if (byteswritten == -1) {//에러라면 -1리턴
		perror("error writing");
		return -1;
	}

	bytesread = n;//위에서 막다룬 bytesread다시 원래값으로 초기화
	bp = buf;//마찬가지로 bp도 원래 값으로 초기화
	printf("\nstarts after 3 seconds...\n");
	sleep(3);

	while (count<n) {

		buf[randomloc[count]] ='1';
		int k = count;//시간정보 저장에 쓰일아이임.
		count++;
		
		printf("\n");
		for(i=0; i<n; i++) {
			printf("%c",buf[i]);
		}

		//고기가 올라간 순간. 여기서부터 시간을재기 시작.
		if (clock_gettime(CLOCK_REALTIME, &tpstart[randomloc[k]]) == -1) { //현재 시간정보를 tpstart[]에 입력해줌
			perror("Failed to get starting time");
			return 1;
		}


		printf("\n뒤집을 고기 선택: ");
		scanf("%d", &b);
		getchar();
		if (b > 0 && b < n + 1) {

			if (buf[b - 1] == '0') {
				printf("뒤집을 고기가 없다.\n");
				printf("+0점");
				done++;
				if(done==n){
					printf("총 점수: %d\n",totalscore);
					break;
				}
			}

			else if (buf[b - 1] =='1') {
				buf[b - 1] ='2';
				for(i=0; i<n; i++) {
					printf("%c",buf[i]);
				}
				if (clock_gettime(CLOCK_REALTIME, &tpend[b - 1]) == -1) {//현재 시각을 tpend[]에 저장
					perror("Failed to get ending time");
					return 1;
				}
				
				timedif = (tpend[b - 1].tv_sec - tpstart[b - 1].tv_sec);//tpstart[]를 현재시각 tpend[]에 빼줌..고기하나하나에 대한 시간을 가져야 하므로 시간담는 변수를 배열로 잡음
				
				if (timedif > 10 && timedif < 20) {//10초에서 20초 사이에 뒤집었으면 perfect
					printf("\nPERFECT +100점\n");
					done++;
					score = 100;
					totalscore += score;
				}
				else if (timedif <= 10 && timedif > 5) {//5초에서 10초사이에 뒤집었으면 EARLY
					printf("\nEARLY +50점\n");
					done++;
					score = 50;
					totalscore += score;
				}
				else if (timedif <= 5 && timedif >= 0) {//0초에서 5초사이에 뒤집었으면 TooEARLY
					printf("\nTooEARLY +0점\n");
					done++;
				}
				else if (timedif >= 20 && timedif < 25) {//20초에서 25초사이에 뒤집었으면 LATE
					printf("\nLATE +50점\n");
					done++;
					score = 50;
					totalscore += score;
				}
				else if (timedif >= 25) {//25초 지나서 뒤집으면 TooLATE
					printf("\nTooLATE +0점\n");
					done++;
				}

				if (done == n) {
					break;//n개 모두 뒤집은게 체크돼면 끝내기
				} else continue;
			}

			else {
				printf("입력 미스.+0점!");
				done++;
				if(done==n){

					break;
				}
			}
			
		}
	}

	printf("총 점수: %d\n",totalscore);
	snprintf(s_totalscore, 10, "%d", totalscore);

	printf("사용자 이름 입력 : ");
	scanf("%s", name);
	strcat(msg, name);
	strcat(msg, ", SCORE : ");
	strcat(msg, s_totalscore);
	strcat(msg, "\n\n");

	fd2 = open("./text/score.txt", O_RDWR | O_APPEND, 0777);
	lseek(fd2, 0, SEEK_END);
	write(fd2, msg, strlen(msg));
	return 0;
}

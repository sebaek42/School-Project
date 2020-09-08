import pandas as pd
import pymysql

x_file = '~/db_score.xlsx'
df = pd.read_excel(x_file)


# 데이터베이스에 연결
connection = pymysql.connect(host='localhost',
                            user='root',
                            password='password',
                            db='university')
try:
    # 커서 생성  pymysql.cursors.DictCursor
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    # column리스트 생성
    cols = "`,`".join([str(i) for i in df.columns.tolist()])

    # df의 행을 한줄씩 넣어줌
    for i,row in df.iterrows():
        sql = "INSERT INTO `db_score` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cursor.execute(sql, tuple(row))
        # connection은 자동커밋이 안되기 때문에 수동으로 해줘야함
        connection.commit()
except:
    print("Database already filled")
finally:
    sql = "SELECT * FROM `db_score`"
    cursor.execute(sql)
    result = cursor.fetchall()
    for i in result:
        print(i)
    cursor.close()
    connection.close()

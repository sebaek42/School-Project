import pymysql

connection = pymysql.connect(host='localhost',
                            user='root',
                            password='password',
                            db='university')
cursor = connection.cursor(pymysql.cursors.DictCursor)

sql = "select sno,midterm, final from db_score"
cursor.execute(sql)

row = cursor.fetchone()
while row:
    if row['midterm'] >= 20 and row['final'] >= 20:
        print("학번: " + str(row['sno']) + " 중간고사: " + str(row['midterm']) + " 기말고사: " + str(row['final']))
    row = cursor.fetchone()
cursor.close()
connection.close()

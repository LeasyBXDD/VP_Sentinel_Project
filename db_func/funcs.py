import mysql.connector

def get_audio_files_from_database():
    conn = mysql.connector.connect(user='admin', password='admin', host='5507', database='audio_database')
    cursor = conn.cursor()
    sql = "SELECT * FROM audios"
    cursor.execute(sql)
    results = cursor.fetchall()
    audio_files = [result[0] for result in results]
    cursor.close()
    conn.close()
    return audio_files

def save_result_to_database(same_speaker_similarity, filename_1, filename_2):
    conn = mysql.connector.connect(user='admin', password='admin', host='5507', database='audio_database')
    cursor = conn.cursor()
    sql = "INSERT INTO audios (audio_file_1, audio_file_2, same_speaker_similarity) VALUES (%s, %s, %s)"
    cursor.execute(sql, (filename_1, filename_2, same_speaker_similarity))
    conn.commit()
    cursor.close()
    conn.close()
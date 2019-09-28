import ARLreader as Ar
import os
import sys
from ftplib import FTP
import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

FTPHost = 'arlftp.arlhq.noaa.gov'   # ftp link of the ARL server
FTPFolder = 'pub/archives/gdas1'   # folder for the GDAS1 data


def fname_from_date(dt):
    """
    parsing the timestamp from the filename and save it into datetime object
    """
    months = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
              7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'}
    week_no = ((dt.day-1)//7)+1
    return 'gdas1.{}{}.w{}'.format(
                                    months[dt.month],
                                    dt.strftime('%y'),
                                    week_no
                                  )


class GDAS1_file_downloader():
    """
    Download the global GDAS1 file from ARL server.
    """

    def __init__(self, saveFolder, bufferSize=10*1024, flagForce=False):
        self.ftpHost = FTPHost
        self.ftpFolder = FTPFolder
        self.buffer = bufferSize
        self.dlSize = 0
        self.flagForce = flagForce
        if os.path.exists(saveFolder):
            self.saveFolder = saveFolder
        else:
            os.mkdir(saveFolder)
            self.saveFolder = saveFolder

    def startConnection(self):
        try:
            self.ftp = FTP(self.ftpHost)
            self.ftp.login()
            print(self.ftp.getwelcome())
            # open the GDAS1 folder
            self.ftp.cwd(self.ftpFolder)
        except Exception as e:
            print('Failure in FTP connection.')
            raise e

    def closeConnection(self):
        self.ftp.close()

    def checkFiles(self, dt):
        """
        check whether the GDAS1 file for the given time, has been saved in
        you local machine.
        """
        file = fname_from_date(dt)
        if os.path.exists(os.path.join(self.saveFolder, file)):
            return True
        else:
            return False

    def download(self, dt):
        """
        Download the GDAS1 file.
        """

        # connect the server
        self.startConnection()

        dlFile = fname_from_date(dt)
        self.dlSize = 0

        # check whether the file exists
        flag = False
        if self.checkFiles(dt) and not self.flagForce:
            print('{file} exists in {folder}.'.format(
                file=dlFile, folder=self.saveFolder))
            return
        else:
            for file in self.ftp.nlst():
                if file == dlFile:
                    flag = True

        lastDisSize = 0

        def dlCallback(data, dlFile, fileSize, f):
            nonlocal lastDisSize
            self.dlSize = self.dlSize + len(data)
            if (self.dlSize - lastDisSize) >= (fileSize * 0.005):
                print('Download {file} {percentage: 04.1f}%'.format(
                    file=dlFile, percentage=(self.dlSize / fileSize) * 100))
                lastDisSize = self.dlSize
            f.write(data)

        if flag:
            self.ftp.sendcmd("TYPE i")    # Switch to Binary mode
            fileSize = self.ftp.size(dlFile)
            self.ftp.sendcmd("TYPE a")    # Swich back to Ascii mode

            with open(os.path.join(self.saveFolder, dlFile), 'wb') as fHandle:
                self.ftp.retrbinary(
                    'RETR {file}'.format(file=dlFile),
                    lambda block: dlCallback(
                                                block,
                                                dlFile,
                                                fileSize,
                                                fHandle
                                            ),
                    blocksize=self.buffer
                    )
        else:
            print('{file} doesn\'t exist in the server.'.format(file=dlFile))

        self.closeConnection()


def extractor(year, month, day, hour, lat, lon, fileNum, *args,
              saveFolder='', globalFolder=''):

    if (not os.path.exists(saveFolder)):
        print('Please set the folder for saving the GDAS1 profiles.')
        raise ValueError

    if (not os.path.exists(globalFolder)):
        print('Please specify the folder of the GDAS1 global files.')
        raise ValueError

    dt = datetime.datetime(year, month, day, hour)

    globalFile = fname_from_date(dt)
    if not os.path.exists(os.path.join(globalFolder, globalFile)):
        print('Start to download {}'.format(globalFile))
        downloader = GDAS1_file_downloader(globalFolder)
        downloader.download(dt)

    profile, sfcdata, indexinfo, ind = Ar.reader(
        os.path.join(
            globalFolder,
            globalFile
        )
    ).load_profile(day, hour, (lat, lon))
    Ar.write_profile(
        os.path.join(saveFolder, str(fileNum)),
        indexinfo,
        ind,
        (lat, lon),
        profile,
        sfcdata
    )


def extractorCity(year, month, day, hour, lat, lon, city, *args,
                  saveFolder='',
                  globalFolder='',
                  force=False):
    """
    extract the GDAS1 profile for a given coordination and time.
    """

    if (not os.path.exists(saveFolder)):
        print('Please set the folder for saving the GDAS1 profiles.')
        raise ValueError

    if (not os.path.exists(globalFolder)):
        print('Please specify the folder of the GDAS1 global files.')
        raise ValueError

    dt = datetime.datetime(year, month, day, hour)

    globalFile = fname_from_date(dt)
    if not os.path.exists(os.path.join(globalFolder, globalFile)):
        print('Global file does not exist. Check {file}.'.
              format(file=globalFile))

        if force:
            print('Start to download {}'.format(globalFile))
            downloader = GDAS1_file_downloader(globalFolder)
            downloader.download(dt)
    else:
        profile, sfcdata, indexinfo, ind = Ar.reader(
                os.path.join(globalFolder, globalFile)
            ).load_profile(day, hour, (lat, lon))
        Ar.write_profile(
            os.path.join(
                            saveFolder,
                            '{city}_{lat:6.2f}_{lon:6.2f}_{date}_{hour}.gdas1'.
                            format(
                                    city=city,
                                    lat=lat,
                                    lon=lon,
                                    date=dt.strftime('%Y%m%d'),
                                    hour=dt.strftime('%H')
                                  )
                        ),
            indexinfo,
            ind,
            (lat, lon),
            profile,
            sfcdata
                        )
        print('Finish writing profie {file}'.
              format(
                  file='{city}_{lat:6.2f}_{lon:6.2f}_{date}_{hour}.gdas1'.
                  format(
                            city=city,
                            lat=lat,
                            lon=lon,
                            date=dt.strftime('%Y%m%d'),
                            hour=dt.strftime('%H')
                        )
                    )
              )


def test():
    # extract gdas1 profile for wuhan
    ###############################################################
    #                        输入                                 #
    ##############################################################
    start = datetime.datetime(2019, 8, 5, 0, 0, 0)
    end = datetime.datetime(2019, 9, 28, 0, 0, 0)
    hours = (end-start).days*24 + (end-start).seconds/3600
    timeList = [start + datetime.timedelta(hours=3*x)
                for x in range(0, int(hours/3))]   # 时间步长每3小时
    saveFolder = 'C:\\Users\\zhenping\\Documents\\Data\\GDAS\\wuhan'
    globalFolder = 'C:\\Users\\zhenping\\Documents\\Data\\GDAS\\global'
    ##############################################################

    for time in timeList:
        extractorCity(
                        time.year,
                        time.month,
                        time.day,
                        time.hour,
                        30.533721,
                        114.367216,
                        'wuhan',
                        saveFolder=saveFolder,
                        globalFolder=globalFolder,
                        force=True
                     )


def main():
    test()


if __name__ == '__main__':
    main()

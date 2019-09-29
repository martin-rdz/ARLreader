import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import ARLreader as Ar
from ftplib import FTP
import argparse
from argparse import RawTextHelpFormatter
import logging
import datetime

FTPHost = 'arlftp.arlhq.noaa.gov'   # ftp link of the ARL server
FTPFolder = 'pub/archives/gdas1'   # folder for the GDAS1 data
LOG_MODE = 'DEBUG'
LOGFILE = 'log'
PROJECTDIR = os.path.dirname(os.path.realpath(__file__))

# initialize the logger
logFile = os.path.join(PROJECTDIR, LOGFILE)
logModeDict = {
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'DEBUG': logging.DEBUG,
    'ERROR': logging.ERROR
    }
logger = logging.getLogger(__name__)
logger.setLevel(logModeDict[LOG_MODE])

fh = logging.FileHandler(logFile)
fh.setLevel(logModeDict[LOG_MODE])
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logModeDict[LOG_MODE])

formatterFh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(funcName)s - %(lineno)d - %(message)s')
formatterCh = logging.Formatter(
    '%(message)s')
fh.setFormatter(formatterFh)
ch.setFormatter(formatterCh)

logger.addHandler(fh)
logger.addHandler(ch)



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
            logger.info(self.ftp.getwelcome())
            # open the GDAS1 folder
            self.ftp.cwd(self.ftpFolder)
        except Exception as e:
            logger.error('Failure in FTP connection.')
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
            logger.warn('{file} exists in {folder}.'.format(
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
                logger.info('Download {file} {percentage: 04.1f}%'.format(
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
            logger.warn('{file} doesn\'t exist in the server.'.format(file=dlFile))

        self.closeConnection()


def extractor(year, month, day, hour, lat, lon, fileNum, *args,
              saveFolder='', globalFolder=''):

    if (not os.path.exists(saveFolder)):
        logger.error('{} does not exist.'.format(saveFolder))
        logger.error('Please set the folder for saving the GDAS1 profiles.')
        raise ValueError

    if (not os.path.exists(globalFolder)):
        logger.error('{} does not exist.'.format(globalFolder))
        logger.error('Please specify the folder of the GDAS1 global files.')
        raise ValueError

    dt = datetime.datetime(year, month, day, hour)

    globalFile = fname_from_date(dt)
    if not os.path.exists(os.path.join(globalFolder, globalFile)):
        logger.info('Start to download {}'.format(globalFile))
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
        logger.error('{} does not exist.'.format(saveFolder))
        logger.error('Please set the folder for saving the GDAS1 profiles.')
        raise ValueError

    if (not os.path.exists(globalFolder)):
        logger.error('{} does not exist.'.format(globalFolder))
        logger.error('Please specify the folder of the GDAS1 global files.')
        raise ValueError

    dt = datetime.datetime(year, month, day, hour)

    globalFile = fname_from_date(dt)
    if not os.path.exists(os.path.join(globalFolder, globalFile)):
        logger.warn('Global file does not exist. Check {file}.'.
              format(file=globalFile))

        if force:
            logger.info('Start to download {}'.format(globalFile))
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
        logger.info('Finish writing profie {file}'.
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



class ArgumentParser(argparse.ArgumentParser):
    """
    Override the error message for argparse.

    reference
    ---------
    https://stackoverflow.com/questions/5943249/python-argparse-and-controlling-overriding-the-exit-status-code/5943381
    """

    def _get_action_from_name(self, name):
        """Given a name, get the Action instance registered with this parser.
        If only it were made available in the ArgumentError object. It is
        passed as it's first arg...
        """
        container = self._actions
        if name is None:
            return None
        for action in container:
            if '/'.join(action.option_strings) == name:
                return action
            elif action.metavar == name:
                return action
            elif action.dest == name:
                return action

    def error(self, message):
        exc = sys.exc_info()[1]
        if exc:
            exc.argument = self._get_action_from_name(exc.argument_name)
            raise exc
        super(ArgumentParser, self).error(message)


def main():

    # Define the command line arguments.
    description = 'extract the GDAS1 profile from GDAS1 global binary data.'
    parser = ArgumentParser(prog='gdas_pro_ext', description=description,
                            formatter_class=RawTextHelpFormatter)

    # Setup the arguments
    helpMsg = "start time of your interests (yyyymmdd-HHMMSS)." +\
              "\ne.g.20151010-120000"
    parser.add_argument("-s", "--start_time",
                        help=helpMsg,
                        dest='start_time',
                        default='20990101-120000')
    helpMsg = "stop time of your interests (yyyymmdd-HHMMSS)." +\
              "\ne.g.20151010-120000"
    parser.add_argument("-e", "--end_time",
                        help=helpMsg,
                        dest='end_time',
                        default='20990101-120000')
    helpMsg = "folder for saving the GDAS1 global files.\n" +\
              "e.g., 'C:\\Users\\zhenping\\Desktop\\global'"
    parser.add_argument("-f", "--global_folder",
                        help=helpMsg,
                        dest='global_folder',
                        default='')
    helpMsg = "folder for saving the extracted profiles\n" +\
              "e.g., 'C:\\Users\\zhenping\\Desktop\\wuhan'"
    parser.add_argument("-o", "--profile_folder",
                        help=helpMsg,
                        dest='profile_folder',
                        default='')

    # if no input arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        # error info can be obtained by using e.argument_name and e.message
        logger.error('Error in parsing the input arguments. Please check ' +
                     'your inputs.\n{message}'.format(message=e.message))
        raise ValueError

    # run the command
    # extract gdas1 profile for wuhan
    ###############################################################
    #                        输入                                 #
    ##############################################################
    start = datetime.datetime.strptime(args.start_time, '%Y%m%d-%H%M%S')
    end = datetime.datetime.strptime(args.end_time, '%Y%m%d-%H%M%S')
    hours = (end-start).days*24 + (end-start).seconds/3600
    timeList = [start + datetime.timedelta(hours=3*x)
                for x in range(0, int(hours/3))]   # 时间步长每3小时
    saveFolder = args.profile_folder
    globalFolder = args.global_folder
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


if __name__ == '__main__':
    main()

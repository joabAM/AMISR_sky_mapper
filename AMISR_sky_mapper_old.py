'''
Created on June 30, 2021

@author: Oper - Joab Apaza
'''

import os
import sys
import glob
import math
import datetime
import time
import argparse
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt


AMISR_MAX_ANGLE = 40.07 #maxima elevacion complementaria a 90
AMISR_BEAM_RESOLUTION = 0.53 #resolucion de apuntes en grados
AMISR_Y_PLANE_RESOLUTION = 0.01325 # 0.53° equivalen a 0.01325 en el plano y con maximo valor = 1

class AMISR_sky_mapper():

    def __init__(self, kwargs):

        '''
        Constructor
        '''

        #threading.Thread.__init__(self)
        self.kwargs = vars(kwargs)
        self.set = None
        self.subset = None
        self.extension_file = '.h5'
        self.dtc_str = 'dtc'
        self.dtc_id = 0
        self.status = True
        self.isConfig = False
        self.dirnameList = []
        self.filenameList = []
        self.fileIndex = None
        self.flagNoMoreFiles = False
        self.flagIsNewFile = 0
        self.filename = ''
        self.amisrFilePointer = None
        self.profileIndex = 0
        self.beamCodeByFrame = None
        self.radacTimeByFrame = None
        self.dataset = None
        self.__firstFile = True
        self.buffer = None
        self.timezone = 'ut'
        self.__waitForNewFile = 20
        self.__filename_online = None
        self.flagNoData = False
        self.realBeamCode = []
        self.azimuth = []
        self.elevation = []
        self.cartesianPoints = []
        self.x_cart = []
        self.y_cart = []
        self.finish = False
        self.minAltIndex = None
        self.maxAltIndex = None
        self.fig = None
        self.radiusXY = {}
        self.configVideo = False
        self.configDir = False
        self.video = None

    def setup(self, **kwargs):

        time_string =   kwargs.get("date")+" "+kwargs.get("startHour")
        self.startTime   =   datetime.datetime.strptime(time_string, "%d/%m/%Y  %H:%M:%S")
        self.path   =   kwargs.get("path")
        time_string =   kwargs.get("date")+" "+kwargs.get("endHour")
        self.endTime    =    datetime.datetime.strptime(time_string, "%d/%m/%Y  %H:%M:%S")
        self.startRange =   kwargs.get("initRange")
        self.endRange   =   kwargs.get("endRange")
        self.flagSingleRange = False
        self.timezone = kwargs.get("timeZone")
        self.factor     = kwargs.get("factor")
        self.mindB      = kwargs.get("minDB")
        self.maxdB      = kwargs.get("maxDB")
        self.fps        = kwargs.get("fps")
        self.dataType   = kwargs.get("dataType")

        if (self.startRange !=None) and (self.endRange == None):
            self.flagSingleRange = True

        if (self.startTime > self.endTime):
            print("Error date and time")
            self.join()

        self.findFiles()

        if not(self.filenameList):
            print("There is no files into the folder: %s"%(path))
            sys.exit(-1)

        self.fileIndex = -1

        self.readNextFile()     #obtiene primeros datos
        cwd = os.getcwd()
        try:
            self.pointings =  np.genfromtxt(cwd+'/UMET_beamcodes.csv', delimiter=',')
        except:
            print("Beamcodes file doesn't exist..")
            #self.join()

        for code in self.beamCode:
            a, e ,x,y = self.decodeAngles(code)


        self.azimuth.sort()
        self.elevation.sort()

        self.x_cart = [n[0] for n in self.cartesianPoints ]
        self.x_cart = list(set(self.x_cart))
        self.x_cart.sort(reverse=False)

        self.y_cart = [n[1] for n in self.cartesianPoints ]
        self.y_cart = list(set(self.y_cart))
        self.y_cart.sort(reverse=True) #
        #print("X points: ",self.x_cart)
        #print("Y points: ",self.y_cart)
        #print(self.elevation)
        self.plot_data_array = np.ones((len(self.y_cart),len(self.x_cart)),dtype=float)

        if not self.getAltitudeIndexes():
            self.flagNoMoreFiles = True #para finalizar el programa
            self.finish = True
            return

        #maximos angulos para los ejes

        theta_x_min = ( math.asin(self.x_cart[0]) * 180/math.pi)  -0.5*AMISR_BEAM_RESOLUTION
        theta_x_max = ( math.asin(self.x_cart[-1]) * 180/math.pi) +0.5*AMISR_BEAM_RESOLUTION
        theta_x_min = round(theta_x_min, 2)
        theta_x_max = round(theta_x_max, 2)

        theta_y_max = ( math.asin(self.y_cart[0]) * 180/math.pi) +0.5*AMISR_BEAM_RESOLUTION
        theta_y_min = ( math.asin(self.y_cart[-1]) * 180/math.pi) -0.5*AMISR_BEAM_RESOLUTION
        theta_y_min = round(theta_y_min, 2)
        theta_y_max = round(theta_y_max, 2)


        list_thetay = np.linspace(theta_y_min, theta_y_max,  len(self.y_cart), endpoint=True) #
        list_thetay = np.round(list_thetay,2)
        #print(self.y_cart[0],len(self.y_cart),theta_y)
        #print(list_thetay)
        #print(self.elevation)
        #print("shapes", self.plot_data_array.shape)
        '''
        methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
        '''
        #print(self.azimuth[0],self.azimuth[-2],self.elevation[0],self.elevation[-1])
        #print(self.y_cart) #puntos eje y
        self.fig = plt.figure(num=55, dpi=100, figsize=(12, 5))
        plt.xlabel('Direction W to E')
        plt.ylabel('Direction S to N')
        self.img = plt.imshow(self.plot_data_array,cmap = 'jet',interpolation='bilinear', aspect='auto',
                    vmin = self.mindB , vmax = self.maxdB , extent=[theta_x_min,theta_x_max,
                    theta_y_min,theta_y_max])

        self.ax = self.fig.gca()
        #self.ax2 = self.ax.twinx()
        self.ax.yaxis.set_ticks(list_thetay)
        #self.ax2.set_ylim(self.y_cart[0],self.y_cart[-1])
        self.ax.grid(which='major', axis='y', linestyle='--', color='k', linewidth=0.3)
        self.ax.grid(which='major', axis='x', linestyle='--', color='k', linewidth=0.3)
        self.ax.minorticks_on()
        #self.ax2.minorticks_on()
        # Adding the colorbar
        self.cbar = self.fig.colorbar(self.img, shrink=0.9)
        self.cbar.minorticks_on()

        plt.ion()

        self.isConfig = True



    def readAMISRHeader(self,fp):
        header = 'Raw11/Data/RadacHeader'
        self.beamCodeByPulse = fp.get(header+'/BeamCode') # LIST OF BEAMS PER PROFILE, TO BE USED ON REARRANGE
        #beamCodes no es confiable en el orden de los canales
        self.frameCount = fp.get(header+'/FrameCount')# NOT USE FOR THIS
        self.modeGroup = fp.get(header+'/ModeGroup')# NOT USE FOR THIS
        self.nsamplesPulse = fp.get(header+'/NSamplesPulse')# TO GET NSA OR USING DATA FOR THAT
        self.npulsesIntegrated = fp.get('/Raw11/Data/PulsesIntegrated')
        self.pulseCount = fp.get(header+'/PulseCount')# NOT USE FOR THIS
        self.radacTime = fp.get(header+'/RadacTime')# 1st TIME ON FILE ANDE CALCULATE THE REST WITH IPP*nindexprofile
        self.timeUnix = fp.get('Time/UnixTime')
        self.timeCount = fp.get(header+'/TimeCount')# NOT USE FOR THIS
        self.timeStatus = fp.get(header+'/TimeStatus')# NOT USE FOR THIS
        self.rangeFromFile = fp.get('Raw11/Data/Samples/Range')
        self.frequency =  fp.get('Rx/Frequency')
        txAus = fp.get('Raw11/Data/Pulsewidth')
        self.experimentFile = fp['Setup/Experimentfile'][()].decode()
        self.beamcodeFile = fp['Setup/Beamcodefile'][()].decode()
        self.trueBeams = self.beamcodeFile.split("\n")
        self.trueBeams.pop()#remove last
        [self.realBeamCode.append(x) for x in self.trueBeams if x not in self.realBeamCode]
        self.realBeamCode = [int(x, 16) for x in self.realBeamCode]
        self.beamCode = self.realBeamCode
        self.experimentName = self.experimentFile[self.experimentFile.find("Name=")+5:self.experimentFile.find("Description=")-2]

        self.nblocks = self.pulseCount.shape[0] #nblocks
        self.nprofiles = self.pulseCount.shape[1] #nprofile
        self.nsa = self.nsamplesPulse[0,0] #ngates
        self.nchannels = len(self.beamCode)
        self.ippSeconds = (self.radacTime[0][1] -self.radacTime[0][0]) #Ipp in seconds
        #self.__waitForNewFile = self.nblocks  # wait depending on the number of blocks since each block is 1 sec
        self.__waitForNewFile = self.nblocks * self.nprofiles * self.ippSeconds # wait until new file is created

        #filling radar controller header parameters
        self.__ippKm = self.ippSeconds *.15*1e6 # in km
        self.__txA = (txAus[()])*.15 #(ipp[us]*.15km/1us) in km
        self.__txB = 0
        nWindows=1
        self.__nSamples = self.nsa
        self.__firstHeight = self.rangeFromFile[0][0]/1000 #in km
        self.__deltaHeight = (self.rangeFromFile[0][1] - self.rangeFromFile[0][0])/1000

        #filling system header parameters
        self.__nSamples = self.nsa
        self.newProfiles = self.nprofiles/self.nchannels
        self.__channelList = list(range(self.nchannels))
        self.__frequency = self.frequency[0][0]


    def removeFiles(self,path):
        files_list = os.listdir(path)
        for file in files_list:
            os.remove(path+file)


    def decodeAngles(self,code):

        r,c = np.where(self.pointings == code)
        #print(r,c)
        r = int(r)
        c = int(c)
        azi = self.pointings[r,1]
        elev = self.pointings[r,2]
        #print(azi,elev)
        #proyecciones en plano cartesiano, 2 dimensiones
        #libreria math usa radianes
        elev_rad = elev/180 * math.pi
        azi_rad = azi/180 * math.pi
        x_cart = round(math.cos(elev_rad)*math.sin(azi_rad),2)
        y_cart = round(math.cos(elev_rad)*math.cos(azi_rad),2)
        r = round(math.sqrt(pow(x_cart,2)+pow(y_cart,2)),3)
        new = [x_cart,y_cart,r]
        if not(new in self.cartesianPoints):
            self.cartesianPoints.append(new)


        if not(azi in self.azimuth):             #it creates the table of angles
            self.azimuth.append(azi)
        if not(elev in self.elevation):
            self.elevation.append(elev)

        return azi, elev, x_cart, y_cart



    def roundPartial (self,value, resolution): #no se usa
        return round(value / resolution) * resolution



    def getAltitudeIndexes(self):

        if self.rangeFromFile == None:
            return False

        if self.flagSingleRange:
            self.endRange=self.startRange

        for n in range(len(self.rangeFromFile[0][:])):

            if (((self.rangeFromFile[0][n]/1000) > self.startRange) and (self.minAltIndex == None)) :
                self.minAltIndex = n


            if (((self.rangeFromFile[0][n]/1000) > self.endRange) and (self.maxAltIndex == None)):
                self.maxAltIndex = n

        if self.minAltIndex == self.minAltIndex:
            self.maxAltIndex = self.maxAltIndex+1

        if (self.minAltIndex==None) or (self.maxAltIndex == None):
            print("Error, altitudes out of range")
            return False

        return True


    def __getTimeFromData(self):
        startDateTime_Reader = self.startTime
        endDateTime_Reader = self.endTime

        print('Filtering Files from %s to %s'%(startDateTime_Reader, endDateTime_Reader))
        print('........................................')
        filter_filenameList = []
        self.filenameList.sort()
        #for i in range(len(self.filenameList)-1):
        for i in range(len(self.filenameList)):
            filename = self.path+ self.filenameList[i]
            fp = h5py.File(filename,'r')
            time_str = fp.get('Time/RadacTimeString')

            startDateTimeStr_File = time_str[0][0].decode('UTF-8').split('.')[0]
            #startDateTimeStr_File = "2019-12-16 09:21:11"
            junk = time.strptime(startDateTimeStr_File, '%Y-%m-%d %H:%M:%S')
            startDateTime_File = datetime.datetime(junk.tm_year,junk.tm_mon,junk.tm_mday,junk.tm_hour, junk.tm_min, junk.tm_sec)

            #endDateTimeStr_File = "2019-12-16 11:10:11"
            endDateTimeStr_File = time_str[-1][-1].decode('UTF-8').split('.')[0]
            junk = time.strptime(endDateTimeStr_File, '%Y-%m-%d %H:%M:%S')
            endDateTime_File = datetime.datetime(junk.tm_year,junk.tm_mon,junk.tm_mday,junk.tm_hour, junk.tm_min, junk.tm_sec)
            fp.close()

            if self.timezone == 'lt':
                startDateTime_File = startDateTime_File - datetime.timedelta(minutes = 300)
                endDateTime_File = endDateTime_File - datetime.timedelta(minutes = 300)
            if (endDateTime_File>=startDateTime_Reader and endDateTime_File<endDateTime_Reader):
                #self.filenameList.remove(filename)
                filter_filenameList.append(filename)

            if (endDateTime_File>=endDateTime_Reader):
                break


        filter_filenameList.sort()
        self.filenameList = filter_filenameList
        return 1

    def __filterByGlob1(self, dirName):
        filter_files = glob.glob1(dirName, '*.*%s'%self.extension_file)
        filter_files.sort()
        filterDict = {}
        filterDict.setdefault(dirName)
        filterDict[dirName] = filter_files
        return filterDict

    def __getFilenameList(self, fileListInKeys, dirList):
        for value in fileListInKeys:
            dirName = list(value.keys())[0]
            for file in value[dirName]:
                filename = os.path.join(dirName, file)
                self.filenameList.append(filename)



    def findFiles(self):
        list =  os.listdir(self.path)      #read all files
        list.sort()
        for file in list:
            if file.endswith(".dt0.h5"):  #dt0 asociado al hardware de AMISR-14
                self.filenameList.append(file)


        self.__getTimeFromData() #filtrar segun fecha y hora

        for i in range(len(self.filenameList)):
            print("%s" %(self.filenameList[i]))
        return


    def __setNextFile(self):
        idFile = self.fileIndex

        while (True):
            idFile += 1
            if not(idFile < len(self.filenameList)):
                self.flagNoMoreFiles = 1
                print("No more Files")
                return 0
            filename = self.filenameList[idFile]
            amisrFilePointer = h5py.File(filename,'r')
            break

        self.flagIsNewFile = 1
        self.fileIndex = idFile
        self.filename = filename
        self.amisrFilePointer = amisrFilePointer
        print("Setting the file: %s"%self.filename)
        return 1


    def readData(self):
        dataset = None
        buffer = None
        if self.dataType == "volts":
            dataset= self.amisrFilePointer.get('Raw11/Data/Samples/Data')
            I = dataset[:,:,:,0]
            Q = dataset[:,:,:,1]
            buffer = (np.power(I,2) + np.power(Q,2)) #debido a raíz de 2 y 50ohm #retorna watts
            dataset = buffer
            #print(dataset[0][0][0])
        else:
            dataset = self.amisrFilePointer.get('Raw11/Data/Power/Data')

        self.radacTime = self.amisrFilePointer.get('Raw11/Data/RadacHeader/RadacTime')
        timeset = self.radacTime[:,0]

        return dataset,timeset


    def readNextFile(self):

        newFile = self.__setNextFile()

        if not(newFile):
            self.error = True
            return 0

        self.readAMISRHeader(self.amisrFilePointer)

        self.dataset,   self.timeset    =   self.readData()

        if self.endTime!=None:
            endDateTime_Reader = self.endTime
            time_str = self.amisrFilePointer.get('Time/RadacTimeString')
            startDateTimeStr_File = time_str[0][0].decode('UTF-8').split('.')[0]
            junk = time.strptime(startDateTimeStr_File, '%Y-%m-%d %H:%M:%S')
            startDateTime_File = datetime.datetime(junk.tm_year,junk.tm_mon,junk.tm_mday,junk.tm_hour, junk.tm_min, junk.tm_sec)

            if self.timezone == 'lt':
                startDateTime_File = startDateTime_File - datetime.timedelta(minutes = 300)
                if (startDateTime_File>endDateTime_Reader):
                    return 0



        self.profileIndex = 0

        return 1


    def __hasNotDataInBuffer(self):
        #if self.profileIndex >= (self.newProfiles*self.nblocks):
        if self.profileIndex >= (self.newProfiles):
            return 1
        return 0

    def _w2DBm(self, power):
        #p = (10 * math.log10(power))
        p = (10 * np.log10(power))
        return p




    def getData(self):

        if self.flagNoMoreFiles:
            self.flagNoData = True
            return 0

        if self.__hasNotDataInBuffer():
            if not (self.readNextFile()):
                return 0


        # if self.dataset is None: # setear esta condicion cuando no hayan datos por leer
        #     self.flagNoData = True
        #     return 0

        #print(self.beamCode)
        if self.dataType == "power":
            beams = self.beamCode   # total de apuntes

        elif self.dataType == "volts":
            beams = self.beamCodeByPulse[self.profileIndex,:] #total de apuntes x integraciones
        #print(beams.shape)
        for altIndex in range(self.minAltIndex, self.maxAltIndex,1):
            i = 0
            prevBeam = beams[0]
            spower = 0
            count = 0
            for beam in beams:
                power = 0
                a, e, x, y  = self.decodeAngles(beam)
                c = self.x_cart.index(x)
                r = self.y_cart.index(y)

                if self.dataType == "power":
                    power = (self.dataset[self.profileIndex,i,altIndex])/(self.npulsesIntegrated[0][0])
                    self.plot_data_array[r,c] = power

                elif self.dataType == "volts":
                    if beam==prevBeam:
                        spower +=  self.dataset[self.profileIndex, i, altIndex]
                        count += 1
                    else:
                        power = spower / count
                        count = 0
                        spower = 0
                        self.plot_data_array[r,c] = power
                prevBeam = beam #otro beam
                i = i+1     #incremento al siguiente beam

            self.plot_data_array = self._w2DBm(self.plot_data_array)
            self.plot_data_array = np.ndarray.round(self.plot_data_array,2)

            date = datetime.datetime.fromtimestamp(int(self.timeUnix[self.profileIndex,0])).strftime('%Y-%m-%d %H:%M:%S')
            altitude = round(self.rangeFromFile[0][altIndex]/1000, 2)

            self.plotData( date, altitude)
            self.plot_data_array = np.ones((len(self.y_cart),len(self.x_cart))) #reset

        self.profileIndex += 1

        return self.profileIndex


    def plotData(self, date , altitude):

        if (plt.gcf().number != 55):
            self.destroyer()
            return

        title = "AMISR " + date + "  at  "+str(altitude)+" km"
        self.img.set_array(self.plot_data_array)
        plt.title(title)

        if self.flagSingleRange:
            outDataName = date+"_"+str(int(self.startRange))
        else:
            outDataName = date+"_"+str(int(self.startRange))+"to"+str(int(self.endRange))

        if not self.configDir:
            cdir = os.getcwd()
            self.wfolder = cdir +"/"+"AMISR CUTS/"+self.experimentName+"/"
            if not os.path.isdir(self.wfolder):
                os.makedirs(self.wfolder)
            self.cutFolder = self.wfolder + outDataName+"/"
            if not os.path.isdir(self.cutFolder): #carpeta de cortes
                os.makedirs(self.cutFolder)
            else:
                self.removeFiles(self.cutFolder) #eliminar los archivos pre-existentes

            text_file = open(self.wfolder+self.experimentName+".txt", "w")
            n = text_file.write(self.experimentFile)
            text_file.close()
            self.configDir = True

        figName = date +"_"+str(int(altitude*100))+'.jpg'
        pathFig = self.cutFolder+figName
        plt.savefig(pathFig)


        if not self.configVideo:
            frame = cv2.imread(pathFig)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, layers = frame.shape
            #video_name = "amisr_"+date[:-9]+".mp4"
            video_name = "amisr_"+outDataName+".mp4"
            self.video = cv2.VideoWriter(self.wfolder+video_name, fourcc, self.fps, (width, height))
            self.configVideo = True

        self.video.write(cv2.imread(pathFig))
        #time.sleep(0.1)
        plt.pause(0.05)

    def plotVoltage(self, date , altitudes ):

        pass

    def plotRTI(self):
        if (plt.gcf().number != 45): #45 figure of RTI
            self.destroyer()
            return
        if not self.configRTI:
            title = "AMISR RTI " + date
            plt.title(title)
        self.img.set_array(self.plot_data_array)


        pass


    def destroyer(self):

        if  self.configVideo:
            self.video.release()
            cv2.destroyAllWindows()
        plt.ioff()
        plt.close('all')
        self.flagNoMoreFiles = True
        self.finish = True



    def run(self):
        '''
        This method will be called many times so here you should put all your code
        '''

        if not self.isConfig:
            self.setup(**self.kwargs)

        if self.finish:
             return
        self.getData()



'''
python3 AMISR_sky_mapper.py --path=/home/soporte/Documentos/AMISR/RAW_DATA/20210707.002/ --date=07/07/2021 --startHour=11:15:00 --endHour=12:00:00 --initRange=0 --endRange=200 --maxDB=75 --dataType=volts 

'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str, default=None, help="hdf5 files location")
    parser.add_argument("--date",type=str, default=None, help="Date DD/MM/YYYY")
    parser.add_argument("--startHour",type=str, default="00:00:00", help="Initial hour HH:MM:SS")
    parser.add_argument("--endHour",type=str, default="23:59:59", help="End hour HH:MM:SS")
    parser.add_argument("--initRange",type=float, default=None, help="first altitude in km")
    parser.add_argument("--endRange",type=float, default=None, help="last altitude in km")
    parser.add_argument("--timeZone",type=str, default='lt', help="date time zone")
    parser.add_argument("--factor",type=float, default='1.0', help="image scale")
    parser.add_argument("--minDB",type=int, default='45', help="min DB power")
    parser.add_argument("--maxDB",type=int, default='60', help="max DB power")
    parser.add_argument("--fps",type=int, default='5', help="frames per second")
    parser.add_argument("--dataType",type=str, default="volts", help="data type read for processing")
    kwargs = parser.parse_args()

    pltAMISR = AMISR_sky_mapper(kwargs)

    while (not pltAMISR.flagNoMoreFiles):
        pltAMISR.run()

    print("finishing program...")
    pltAMISR.destroyer()
    plt.ioff()

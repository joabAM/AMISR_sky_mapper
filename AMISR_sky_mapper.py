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
AMISR_PROFILE_ROW_MAX = 3
class PlotPowerCuts():
    def __init__(self, x, y, data, minDB=40, maxDB=75, dataType="raw",expName = "null", initAlt=0,
                    endAlt=300, single=False, path="",pathVideo="" ,outName = "", fps=5):
        #maximos angulos para los ejes
        self.x = x
        self.y = y
        self.data_array = data #matriz de datos, vacia al iniciar
        self.configVideo = False
        self.dataType = dataType
        self.video = None
        self.minDB = minDB
        self.maxDB = maxDB
        self.initAlt = initAlt
        self.endAlt = endAlt
        self.experimentName = expName
        self.singleAlt = single
        self.fig = None
        self.outPath = path
        self.videoPath =pathVideo
        self.outName = outName
        self.fps = fps

    def setup(self):
        theta_x_min = ( math.asin(self.x[0]) * 180/math.pi)  -0.5*AMISR_BEAM_RESOLUTION
        theta_x_max = ( math.asin(self.x[-1]) * 180/math.pi) +0.5*AMISR_BEAM_RESOLUTION
        self.theta_x_min = round(theta_x_min, 2)
        self.theta_x_max = round(theta_x_max, 2)

        theta_y_max = ( math.asin(self.y[0]) * 180/math.pi) +0.5*AMISR_BEAM_RESOLUTION
        theta_y_min = ( math.asin(self.y[-1]) * 180/math.pi) -0.5*AMISR_BEAM_RESOLUTION
        theta_y_min = round(theta_y_min, 2)
        theta_y_max = round(theta_y_max, 2)


        list_thetay = np.linspace(theta_y_min, theta_y_max,  len(self.y), endpoint=True) #
        list_thetay = np.round(list_thetay,2)

        '''
        methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
        '''

        self.fig = plt.figure(num=55, dpi=100, figsize=(12, 5))
        plt.xlabel('Direction W to E')
        plt.ylabel('Direction S to N')
        self.img = plt.imshow(self.data_array,cmap = 'jet',interpolation='bilinear', aspect='auto',
                    vmin = self.minDB , vmax = self.maxDB , extent=[self.theta_x_min,self.theta_x_max,
                    theta_y_min,theta_y_max])

        self.ax = self.fig.gca()
        self.ax2 = self.ax.twiny()

        self.ax.yaxis.set_ticks(list_thetay)
        #self.ax2.set_ylim(self.y_cart[0],self.y_cart[-1])
        self.ax.grid(which='major', axis='y', linestyle='--', color='k', linewidth=0.3)
        self.ax.grid(which='major', axis='x', linestyle='--', color='k', linewidth=0.3)
        self.ax.minorticks_on()
        #self.ax2.minorticks_on()
        # Adding the colorbar
        self.cbar = self.fig.colorbar(self.img, shrink=0.9)
        self.cbar.minorticks_on()

        # circle1 = plt.Circle((0, 0), 2, color='w', fill=False,linestyle="--")
        # circle2 = plt.Circle((0, 0), 6, color='w', fill=False,linestyle="--")
        # circle3 = plt.Circle((0, 0), 10, color='w', fill=False,linestyle="--")
        # circle4 = plt.Circle((0, 0), 18, color='w', fill=False,linestyle="--")
        # self.ax.add_patch(circle1)
        # self.ax.add_patch(circle2)
        # self.ax.add_patch(circle3)
        # self.ax.add_patch(circle4)
        plt.ion()
        return True


    def plotData(self, date , altitude, array_data):
        self.data_array = array_data
        if (plt.gcf().number != 55):
            self.destroy()
            return False

        title = "AMISR " + date + "  at  "+str(altitude)+" km"
        self.img.set_array(self.data_array)
        plt.title(title)

        x_l, x_r = self.ax.get_xlim()
        ticks = self.ax2.get_xticklabels()
        n = math.ceil(len(ticks)/2)
        n_l = math.cos(self.theta_x_min*math.pi/180)*altitude
        n_r = math.cos(self.theta_x_max*math.pi/180)*altitude
        label1 = np.linspace(n_l,altitude,n)
        label2 = np.linspace(altitude,n_r,n)
        labels = np.append(label1.astype(int),label2.astype(int))
        ticklabels = [str(item)+" km" for item in labels]
        self.ax2.set_xticklabels(ticklabels)

        '''
        Clase padre de gráficos pendiente
        '''


        figName = date +"_"+str(int(altitude*100))+'.jpg'
        pathFig = self.outPath+figName
        plt.savefig(pathFig)


        if not self.configVideo:
            frame = cv2.imread(pathFig)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, layers = frame.shape
            #video_name = "amisr_"+date[:-9]+".mp4"
            video_name = "amisr_"+self.outName+"_"+self.dataType+".mp4"
            self.video = cv2.VideoWriter(self.videoPath+video_name, fourcc, self.fps, (width, height))
            self.configVideo = True

        self.video.write(cv2.imread(pathFig))
        #time.sleep(0.1)
        plt.pause(0.05)
        return True

    def destroy(self):

        if  self.configVideo:
            self.video.release()
            cv2.destroyAllWindows()
        plt.ioff()
        plt.close('all')

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
class PlotPowerProfiles():
    def __init__(self,channelList,altitudes, minDB=40, maxDB=75, expName = "null",
                  path=""):
        self.channelList = channelList
        self.nchannels = len(channelList)
        self.altitudes = altitudes/1000  #a km
        self.minDB = minDB
        self.maxDB = maxDB
        self.expName = expName
        self.outPath = path+"profiles/"
        self.fig = None
        self.axs = None
        pass

    def setup(self):
        if self.nchannels == 1:
            self.fig, self.axs = plt.subplots(num=45)
        elif self.nchannels <= AMISR_PROFILE_ROW_MAX:
            self.fig, self.axs = plt.subplots(self.nchannels, num=45)
        else:
            c = math.ceil(self.nchannels/AMISR_PROFILE_ROW_MAX)
            self.fig, self.axs = plt.subplots(AMISR_PROFILE_ROW_MAX,c, num=45)

        if not os.path.isdir(self.outPath): #carpeta de perfiles
            os.makedirs(self.outPath)
        self.axs.grid(which='major', axis='both')
        self.axs.grid(which='minor', axis='y', linestyle='--')
        self.axs.minorticks_on()

    def plotProfiles(self, date, data):

        if (plt.gcf().number != 45):
            self.destroy()
            return False
        if self.nchannels == 1:
            for p in range(self.nchannels):
                self.axs.set_title("Beam :"+str(self.channelList))
                x = data[:,p].tolist()
                self.axs.plot(x,self.altitudes)

        elif self.nchannels>AMISR_PROFILE_ROW_MAX:
            rows, cols = self.axs.shape
            i = 0
            for r in range(rows):
                for c in range(cols):
                    self.axs[r,c].set_title("Beam :"+str(self.channelList[i]))
                    x = data[:,i].tolist()

                    self.axs[r,c].plot(x,self.altitudes)
                    i += 1
        else:
            for p in range(self.nchannels):
                self.axs[p].set_title("Beam :"+str(self.channelList))
                x = data[:,p].tolist()
                self.axs[p].plot(x,self.altitudes)

        figName = date +"_profile"+'.jpg'
        pathFig = self.outPath+figName
        plt.show(block=False)
        plt.savefig(pathFig)
        return True

    def destroy(self):

        plt.close('all')
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
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
        self.configDir = False
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
        self.radiusXY = {}
        self.outDataName =""
        self.c_date = None #current date
        self.plot = None


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
        self.dataInType   = kwargs.get("dataInType")
        self.plotType   = kwargs.get("plotType")
        chans = kwargs.get("channels").split(",")
        self.channelList = [int(n) for n in chans]
        ########################################################################

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

        self.readNextFile()     #obtiene primeros datos del header
        cwd = os.getcwd()
        try:
            self.pointings =  np.genfromtxt(cwd+'/UMET_beamcodes.csv', delimiter=',')
        except:
            print("Beamcodes file doesn't exist..")
            #self.join()

        for code in self.beamCode:
            a, e ,x,y = self.decodeAngles(code)

        ########################################################################
        self.azimuth.sort()
        self.elevation.sort()

        self.x_cart = [n[0] for n in self.cartesianPoints ]
        self.x_cart = list(set(self.x_cart))
        self.x_cart.sort(reverse=False)

        self.y_cart = [n[1] for n in self.cartesianPoints ]
        self.y_cart = list(set(self.y_cart))
        self.y_cart.sort(reverse=True) #

        self.plot_data_array = np.ones((len(self.y_cart),len(self.x_cart)),dtype=float)

        ########################################################################

        if not self.getAltitudeIndexes():
            self.flagNoMoreFiles = True #para finalizar el programa
            self.finish = True
            return
        self.naltitudes =  self.maxAltIndex - self.minAltIndex
        self.plot_data =  np.ones((self.naltitudes ,len(self.channelList)))

        self.getData()  #para configurar fechas y carpetas
        self.profileIndex = 0 #para que no pierda el primer perfil afecta la lectura
        ########################################################################

        if self.flagSingleRange:
            self.outDataName = self.c_date+"_"+str(int(self.startRange))
        else:
            self.outDataName = self.c_date+"_"+str(int(self.startRange))+"to"+str(int(self.endRange))

        if not self.configDir:
            cdir = os.getcwd()
            self.wfolder = cdir +"/"+"AMISR CUTS/"+self.experimentName+"/"
            if not os.path.isdir(self.wfolder):
                os.makedirs(self.wfolder)
            self.cutFolder = self.wfolder + self.outDataName+"_"+self.dataInType+"/"
            if not os.path.isdir(self.cutFolder): #carpeta de cortes
                os.makedirs(self.cutFolder)

            else:
                self.removeFiles(self.cutFolder) #eliminar los archivos pre-existentes

            text_file = open(self.wfolder+self.experimentName+".txt", "w")
            n = text_file.write(self.experimentFile)
            text_file.close()
            self.configDir = True

        ########################################################################

        if self.plotType == "cuts":
            self.plot = PlotPowerCuts(self.x_cart, self.y_cart, self.plot_data_array,pathVideo=self.wfolder,dataType = self.dataInType,
                        expName=self.experimentName, initAlt=self.startRange, endAlt=self.endRange,minDB=self.mindB,
                        maxDB=self.maxdB, single=self.flagSingleRange, path=self.cutFolder, outName=self.outDataName)
            self.plot.setup()
        elif self.plotType == "profiles":
            self.plot = PlotPowerProfiles(self.channelList,self.rangeFromFile[0,self.minAltIndex:self.maxAltIndex],
                        minDB=self.mindB, maxDB=self.maxdB, path=self.cutFolder)
            self.plot.setup()
            pass

        ########################################################################
        self.isConfig = True



    def readAMISRHeader(self,fp):
        header = 'Raw11/Data/RadacHeader'
        self.beamCodeByPulse = fp.get(header+'/BeamCode') # LIST OF BEAMS PER PROFILE, TO BE USED ON REARRANGE
        print(self.beamCodeByPulse)
        self.realBeamCode = [] # beamCode limpio
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
        if self.dataInType == "coded":
            self.rangeFromFile = fp.get('CohCode/Data/Power/Range')
        else:
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
            if os.path.isdir(path+file):
                self.removeFiles(path+file+"/")
                os.rmdir(path+file)
            else:
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
        if self.dataInType == "volts":
            dataset= self.amisrFilePointer.get('Raw11/Data/Samples/Data')
            I = dataset[:,:,:,0]
            Q = dataset[:,:,:,1]
            buffer = (np.power(I,2) + np.power(Q,2)) #debido a raíz de 2 y 50ohm #retorna watts
            dataset = buffer

        elif self.dataInType =="coded":
            dataset = self.amisrFilePointer.get('CohCode/Data/Power/Data')
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


        if self.dataInType == "power" or self.dataInType=="coded":
            beams = self.beamCode  # total de apuntes

        elif self.dataInType == "volts":
            beams = self.beamCodeByPulse[self.profileIndex,:] #total de apuntes x integraciones
        #print(beams.shape)
        n = 0

        a, e, x, y, r, c = [0,0,0,0,0,0]
        for altIndex in range(self.minAltIndex, self.maxAltIndex,1):
            i = 0
            k = 0
            prevBeam = beams[0]
            spower = 0
            count = 0
            for beam in beams:
                power = 0
                if self.plotType=="cuts":
                    a, e, x, y  = self.decodeAngles(beam)
                    c = self.x_cart.index(x)
                    r = self.y_cart.index(y)

                if self.dataInType == "power" or self.dataInType=="coded":
                    power = (self.dataset[self.profileIndex,i,altIndex])/(self.npulsesIntegrated[0][0])
                    self.plot_data_array[r,c] = power
                    if i in self.channelList and self.plotType=="profiles":
                        #print(n,k,i)
                        self.plot_data[n,k] = power
                        k += 1
                elif self.dataInType == "volts":
                    if beam==prevBeam:
                        spower +=  self.dataset[self.profileIndex, i, altIndex]
                        count += 1
                    else:
                        power = spower / count
                        count = 0
                        spower = 0
                        self.plot_data_array[r,c] = power
                        if i in self.channelList and self.plotType=="profiles":
                            self.plot_data[n,k] = power
                            k += 1
                prevBeam = beam #otro beam
                i = i+1     #incremento al siguiente beam
            n += 1 #incrementro por cada altura

            self.plot_data_array = self._w2DBm(self.plot_data_array)
            self.plot_data_array = np.ndarray.round(self.plot_data_array,2)
            c_date = datetime.datetime.fromtimestamp(int(self.timeUnix[self.profileIndex,0])).strftime('%Y-%m-%d %H:%M:%S')
            c_altitude = round(self.rangeFromFile[0][altIndex]/1000, 2)
            self.c_date = c_date

            if self.isConfig and self.plotType=="cuts":
                if not self.plot.plotData( c_date, c_altitude, self.plot_data_array):
                    self.destroyer()
            self.plot_data_array = np.ones((len(self.y_cart),len(self.x_cart))) #reset the array

        if self.isConfig and self.plotType=="profiles":
            self.plot_data = self._w2DBm(self.plot_data)
            self.plot_data = np.ndarray.round(self.plot_data,2)
            if not (self.plot.plotProfiles( c_date, self.plot_data)):
                self.destroyer()
            self.plot_data =  np.ones((self.naltitudes ,len(self.channelList)))  #reset the array

        self.profileIndex += 1

        return self.profileIndex



    def destroyer(self):

        self.plot.destroy()
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
python3 AMISR_sky_mapper.py --path=/home/soporte/Documentos/AMISR/RAW_DATA/20210707.002/ --date=07/07/2021 --startHour=11:15:00 --endHour=12:00:00 --initRange=0 --endRange=200 --maxDB=75 --dataInType=volts --plotType=cuts
python3 AMISR_sky_mapper.py --path=/home/soporte/Documentos/AMISR/RAW_DATA/20210707.002/ --date=07/07/2021 --startHour=11:15:00 --endHour=12:00:00 --initRange=-10 --endRange=50 --maxDB=75 --dataInType=power --plotType=profiles --channels='0'

'''
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str, default=None, help="hdf5 files location")
    parser.add_argument("--date",type=str, default=None, help="Date DD/MM/YYYY")
    parser.add_argument("--startHour",type=str, default="00:00:00", help="Initial hour HH/MM/SS")
    parser.add_argument("--endHour",type=str, default="23:59:59", help="End hour HH/MM/SS")
    parser.add_argument("--initRange",type=float, default=None, help="first altitude in km")
    parser.add_argument("--endRange",type=float, default=None, help="last altitude in km")
    parser.add_argument("--timeZone",type=str, default='lt', help="date time zone")
    parser.add_argument("--factor",type=float, default='1.0', help="image scale")
    parser.add_argument("--minDB",type=int, default='45', help="min DB power")
    parser.add_argument("--maxDB",type=int, default='60', help="max DB power")
    parser.add_argument("--fps",type=int, default='5', help="frames per second")
    parser.add_argument("--dataInType",type=str, default="power", help="input data type read for processing, power, volts, coded")
    parser.add_argument("--plotType",type=str, default="cuts", help=" type of plot")
    parser.add_argument("--channels",type=str, default="0,1,2,3,4" ,help="selected channels for profiles and rti")
    kwargs = parser.parse_args()

    pltAMISR = AMISR_sky_mapper(kwargs)
  


    while (not pltAMISR.flagNoMoreFiles):
        pltAMISR.run()

    print("finishing program...")
    pltAMISR.destroyer()
    plt.ioff()

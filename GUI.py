#!/usr/bin/env python

## Graptor - Graz Application for Tomographic Reconstruction
##
## Copyright (C) 2019 Richard Huber, Martin Holler, Kristian Bredies
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.
##
##
## GUI.py:
## Graphical User Interface for tomography reconstruction using TGV/TV
## regularization with frob/sep/nuc norm coupling and Kullback-Leibler
## or L2 fidelity.
##
## -------------------------
## Richard Huber (richard.huber@uni-graz.at)
## Martin Holler (martin.holler@uni-graz.at)
## Kristian Bredies (kristian.bredies@uni-graz.at)
## 
## 21.02.2019
## -------------------------
## If you consider this code to be useful, please cite:
## 
## [1] R. Huber, G. Haberfehlner, M. Holler, G. Kothleitner,
##     K. Bredies. Total Generalized Variation regularization for
##     multi-modal electron tomography. RSC Nanoscale, accepted
##     January 2019.
##
## [2] M. Holler, R. Huber, F. Knoll. Coupled regularization with
##     multiple data discrepancies. Inverse Problems, Special
##     issue on joint reconstruction and multi-modality/multi-spectral
##     imaging, 34(8):084003, 2018.


from __future__ import print_function

import itertools
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import sys
import mrcfile
import os
from matplotlib.pyplot import *
import subprocess
import shlex
import shutil
import glob 
try:##Python2
    from Tkinter import *
    import Tkinter as tk
    import Tkinter, tkFileDialog
    import tkMessageBox
    import tkFont

except ImportError:##Python3
    from tkinter import *
    import tkinter as tk
    from tkinter import messagebox as tkMessageBox
    from tkinter import filedialog as tkFileDialog
    import tkinter.font as tkFont
    
import math
import h5utilities as h5
import random
import time 
import threading;
try:##Python3
	import queue 
except ImportError:##Python2
	import Queue as queue
    
def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)
	
def replace_ext(fname, new_ext):
	(name, ext) = os.path.splitext(fname)
	return name+"."+new_ext


##Function is called if GUI is closed. Ensures that all processes are closed
def ask_quit():
	#if tkMessageBox.askokcancel("Quit", "You want to quit now? All currently running reconstructions will be aborted!"):
	global initial_computationbutton
	request=0
	current_computationbutton=initial_computationbutton
	
	while current_computationbutton.successor!=None:
		if current_computationbutton.success==0:#Process is still running
			#print (current_computationbutton.namefield.get(),current_computationbutton.success)
			request=1
		current_computationbutton=current_computationbutton.successor
	#print(request)
	if request==1:
		if tkMessageBox.askokcancel("Quit", "Do you want to quit now? All currently running reconstructions will be aborted!"):	 
			while initial_computationbutton.successor!=None:
				computationbuttonnew=initial_computationbutton.successor			
				try:
					initial_computationbutton.stoppingbut.Stopp()
				except:
					print('an error occured in ', initial_computationbutton.namefield.get())
					pass
				print ('successfully_destroyed ', initial_computationbutton.namefield.get())
				initial_computationbutton=computationbuttonnew
				
				#close('all')
				#time.sleep(3)
				os._exit(1)
	elif request==0:
		os._exit(1)
#Fenster Erstellen
fenster=Tk()
fenster.title('Graptor Reconstruction Tool')
fenster.option_add("*Font","Times 10 bold" )
fenster.protocol("WM_DELETE_WINDOW", ask_quit)

windowFont = tkFont.Font(family='Times', size=10, weight='bold')
headingFont = tkFont.Font(family='Verdana', size=12, weight='bold')

vis_lower_percentile = 0.1
vis_upper_percentile = 99.9
Infilelist=[]


#
class Outline:
	def __init__(self,master):
		frame=Frame(master)
		frame.grid(column=0,row=0)
		 
		self.label=Label(frame,text='Output file prefix:')
		self.label.pack(side=LEFT)
		
		self.entryfield=Entry(frame);
		self.entryfield.pack(side=LEFT)
		self.entryfield.insert(0,'results/recon')
 
		
		self.frame=frame
		
		self.browsbutton=Button(frame,text='Browse',command=self.browspath)
		self.browsbutton.pack(side=LEFT)
 
		
	def browspath(self):
		try:
			fenster.filename = tkFileDialog.asksaveasfilename(title = "Select file",filetypes = (("all files","*.*"),("MRC files","*.mrc")))
			if fenster.filename!='':
				self.entryfield.delete(0, END)
				self.entryfield.insert(0,os.path.relpath(fenster.filename))
		except:
			pass

### Class concerning widget to enter a file containing bad projections
class Bad_file:
	def __init__(self,master,pos,text1,text2):
		self.frame=Frame(master)
		self.frame.grid(row=pos,column=0,sticky='W')
		self.var = IntVar()
		self.button= Checkbutton(self.frame, text=text1, variable=self.var,command=self.show_input)
		self.button.grid(row=pos,column=0,sticky='W')
		self.pos=pos
		self.hiddenframe=Frame(self.frame)
		self.hiddenframe.grid(row=pos,column=1,sticky='W')
		Label(self.hiddenframe,text=text2).pack(side=LEFT)
		self.entryfield=Entry(self.hiddenframe,width=30)
		self.entryfield.pack(side=LEFT)
		self.browsbutton=Button(self.hiddenframe,text='Browse',command=self.browse)
		self.browsbutton.pack()
		self.hiddenframe.grid_remove()
	def show_input(self):
		
		if self.var.get()==0:
			self.hiddenframe.grid_remove()
		else:		
			self.hiddenframe.grid(row=self.pos,column=1)
	def browse(self):
		try:
			global startingpath
			fenster.filename = tkFileDialog.askopenfilename(initialdir=startingpath,title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
			
			if fenster.filename!='':
				startingpath= '/'.join( fenster.filename.split('/')[:-1] )
				self.entryfield.delete(0, END)
				self.entryfield.insert(0,os.path.relpath(fenster.filename))
		except:
			pass


###Irrelevant Bad search widget class
class Bad_Search:
	def __init__(self,master,pos,text1,text2):

		self.var = IntVar()
		self.button= Checkbutton(master, text=text1, variable=self.var,command=self.show_input)
		self.button.grid(row=pos,column=0,sticky='E')
		self.pos=pos
		self.hiddenframe=Frame(master)
		self.hiddenframe.grid(row=pos,column=1,sticky='W')
		Label(self.hiddenframe,text=text2).pack(side=LEFT)
		self.entryfield1=Entry(self.hiddenframe,width=15)
		self.entryfield1.pack(side=LEFT)
		self.entryfield2=Entry(self.hiddenframe,width=15)
		self.entryfield2.pack(side=LEFT)
		self.hiddenframe.grid_remove()
	def show_input(self):
		if self.var.get()==0:
			self.hiddenframe.grid_remove()
		else:		
			self.hiddenframe.grid(row=self.pos,column=1)

			

###Widget of input of Bad projections
class Badprojections:
	def __init__(self,master,pos,text1,text2):
		self.var = IntVar()
		self.frame=Frame(master)
		self.frame.grid(row=pos,column=0,columnspan=2,sticky='N,W')
		self.button= Checkbutton(self.frame, text=text1, variable=self.var,command=self.show_input)
		self.button.grid(row=pos,column=0,sticky='N,W')
		self.pos=pos
		self.hiddenframe=Frame(self.frame,relief="sunken",borderwidth=1)
		self.hiddenframe.grid(row=pos,column=1,sticky='W')

		self.Firstbutton=Button_with_fields(self.hiddenframe,0,text2[0],'',0,'')
		self.Secondbutton=Bad_file(self.hiddenframe,1,text2[1],'')
		#self.Thirdbutton=Bad_Search(self.hiddenframe,2,text2[2],'')
		self.hiddenframe.grid_remove()
	def show_input(self):
		
		if self.var.get()==0:
			self.hiddenframe.grid_remove()
		else:		
			self.hiddenframe.grid(row=self.pos,column=1)


###Widget class of a button activating a field to be written in
class Button_with_fields:
	def __init__(self,master,pos,text1,text2,state,default):

		self.var = IntVar() 
		self.frame=Frame(master) 
		self.frame.grid(row=pos,column=0,sticky='W') 
		self.button=Checkbutton(self.frame, text=text1, 
		variable=self.var,command=self.show_input) 
		self.button.grid(row=pos,column=0,sticky='W') 
		self.pos=pos 
		self.hiddenframe=Frame(self.frame) 
		self.hiddenframe.grid(row=pos,column=1,sticky='W') 
		Label(self.hiddenframe,text=text2).pack(side=LEFT) 
		self.entryfield=Entry(self.hiddenframe,width=30) 
		self.entryfield.pack(side=LEFT) 
		self.entryfield.insert(0,default) 
		if state==0:
			self.hiddenframe.grid_remove()
		else:
			self.button.select()
	def show_input(self):
		
		if self.var.get()==0:
			self.hiddenframe.grid_remove()
		else:		
			self.hiddenframe.grid(row=self.pos,column=1)

	
###Widget class with a button for Brightness correction options
class Brightness:
	def __init__(self,master,pos):
		self.var = IntVar()
		self.button= Checkbutton(master, text="Perform brightness correction", variable=self.var)
		self.button.grid(row=pos,column=0,sticky='W')
		self.button.select()
		
	


			
			
################## Hoverinformation #################			
class ToolTipBase: 
    def __init__(self, button,size,position=[20,1]): 
     self.button = button 
     self.tipwindow = None 
     self.position=position
     self.id = None 
     self.size=size
     self.x = self.y = 0 
     self._id1 = self.button.bind("<Enter>", self.enter) 
     self._id2 = self.button.bind("<Leave>", self.leave) 
     self._id3 = self.button.bind("<Button-1>", self.leave) 

    def enter(self, event=None): 
     self.schedule() 

    def leave(self, event=None): 
     self.unschedule() 
     self.hidetip() 

    def schedule(self): 
     self.unschedule() 
     self.id = self.button.after(1500, self.showtip) 

    def unschedule(self): 
     id = self.id 
     self.id = None 
     if id: 
      self.button.after_cancel(id) 

    def showtip(self): 
     if self.tipwindow: 
      return 
     # The tip window must be completely outside the button; 
     # otherwise when the mouse enters the tip window we get 
     # a leave event and it disappears, and then we get an enter 
     # event and it reappears, and so on forever :-(
     x = self.button.winfo_rootx() + self.position[0] 
     y = self.button.winfo_rooty() + self.button.winfo_height() + self.position[1] 
     self.tipwindow = tw = Toplevel(self.button) 
     tw.wm_overrideredirect(1) 
     
     tw.geometry("%dx%d%+d%+d" % (self.size[0], self.size[1], x, y))
     
     self.showcontents() 

    def showcontents(self, text="Your text here"): 
     # Override this in derived class 
     label = Text(self.tipwindow, text=text, justify=LEFT, 
         background="#ffffe0", relief=SOLID, borderwidth=2,
         fill=BOTH, wrap=WORD)
     label.pack() 

    def hidetip(self): 
     tw = self.tipwindow 
     self.tipwindow = None 
     if tw: 
      tw.destroy() 


class ToolTip(ToolTipBase): 

    def __init__(self, button, text): 
     ToolTipBase.__init__(self, button) 
     self.text = text 

    def showcontents(self): 
     ToolTipBase.showcontents(self, self.text) 


class ListboxToolTip(ToolTipBase): 

    def __init__(self, button, items,size,position=[20,1]): 
     ToolTipBase.__init__(self, button,size,position) 
     self.items = items 
     self.size=size
     self.position=position
    def showcontents(self): 
     listbox = Text(self.tipwindow, background="#ffffe0", wrap=WORD) 
     listbox.pack(fill=BOTH) 
     for item in self.items: 
      listbox.insert(END, item) 
      
#############################################

###Widget depicting the slice information
class Slicereader:
	def __init__(self,master,pos,Title):

		frame=Frame(master)
		frame.grid(row=pos,column=1)
		Label(frame,text=Title).grid(row=pos,column=0,sticky='W')

		self.frame=frame
		frame2=Frame(self.frame)
		self.frame2=frame2
		frame2.grid(row=pos,column=1)
		Label(frame2,text='Start').grid(row=pos,column=1)
		Label(frame2,text='End').grid(row=pos,column=2)
		Label(frame2,text='Step size').grid(row=pos,column=3)
		self.start=Entry(frame2,width=10)
		self.end=Entry(frame2,width=10)
		self.step=Entry(frame2,width=10)
		self.step.insert(0,1)
		self.start.grid(row=pos+1,column=1)
		self.end.grid(row=pos+1,column=2)
		self.step.grid(row=pos+1,column=3)
		
###Widget Class for plot and overlay information, depicting a button and a field
class Checkbuttons:
	def __init__(self,master,position,verticalpos,title1,title2,default):
		self.frame=Frame(master)
		self.frame.grid(row=position,column=verticalpos,sticky='E')
		self.var = IntVar()
		self.button= Checkbutton(self.frame, text=title1, variable=self.var)
		self.vpos=verticalpos
		self.button.grid(row=position,column=verticalpos,sticky='E')
		self.pos=position
		self.hiddenframe=Frame(self.frame)
		self.hiddenframe.grid(row=position,column=verticalpos+1,sticky='W')
		Label(self.hiddenframe,text= title2).pack(side=LEFT)
		self.entryfield=Entry(self.hiddenframe,width=10)
		self.entryfield.pack(side=LEFT)
		self.entryfield.insert(0,default)
		#self.hiddenframe.grid_remove()
		self.button.select()
	#def show_input(self):
	#	if self.var.get()==0:
	#		self.hiddenframe.grid_remove()
	#	else:		
	#		self.hiddenframe.grid(row=self.pos,column=self.vpos+1)
			



###Function using mean in data to propose weights
def proposed_weightratios():
	means=[]
	for i in range(len(Infilelist)):
		try:
			name=Infilelist[i].entryfield.get()
			[data,angles]=h5.readh5(name)
			Infilelist[i].data=data
			Infilelist[i].angles=angles
			Infilelist[i].projection_number.set(str(0))
			Infilelist[i].name_old=Infilelist[i].entryfield.get()
		except IOError as err:
			template = "An exception of type {0} occurred. Arguments:\n{1!r}"
			message = template.format(type(err).__name__, err.args)
			eprint(message)

			Returnwindow = Toplevel()
			Returnwindow.geometry("")
			Infilelist[i].Returnwindow=Returnwindow
			Infilelist[i].Returnwindow.title('Line number '+str(i)+' can not be read')		
			Infilelist[i].button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy,width=50)
			Infilelist[i].button.pack()
		means.append(np.mean(data))
	means=np.array(means)
	means/=np.max(means)
	
	
	for i in range(len(Infilelist)):
		value=int(means[i]*10**int(-np.log10(means[i])+4))/10.**int(-np.log10(means[i])+4)
		Infilelist[i].weight.delete(0,END)
		Infilelist[i].weight.insert(0,str(value))
	
		
	
### Widget Class for line of entering data
class Inputline:
	def __init__(self,master):
		self.master=master
		frame=Frame(master)
		frame.grid()
		tip = ListboxToolTip(frame, ["Enter the file name you want to read projection data from. The input image data are required to be in MRC format. Additionally, a file containing the relevant angles is needed, either a text file (with same name ending with .rawtlt) containing one angle each line or a CSV file (with same name ending in .csv) separated by ','. The angles are supposed to be given in degree (0-360). The channel name allows to identify the channel easily during the reconstruction process. The weights are used to give a suitable amount of regularization (smoothing) to each channel, see also automated weight choice and the manual for tips on finding suitable parameters. The view option (+/- buttons) allows you to browse through the data."],[550,170]) 

		self.frame=frame
		self.entryfield=Entry(frame,width=30);
		self.entryfield.grid(row=1,column=1)
		
		self.browsbutton=Button(frame,text='Browse',command=self.browspath)
		self.browsbutton.grid(row=1,column=2)

		self.Channelname=Entry(frame)
		self.Channelname.grid(row=1,column=3)
		
		self.weight=Entry(frame,width=13)
		self.weight.grid(row=1,column=4)
		self.weight.insert(0, "0.01")
						
		self.viewminus=Button(frame,text='-', command=self.decrease_projectionnumber)
		self.viewminus.grid(row=1, column=5)				
		
		self.name_old=''
		self.figure_number=-1
		self.projection_number=StringVar()
		self.projection_number.set('0')
		self.angles=[1]
		self.projection_number_field=Entry(frame,width=4,text=self.projection_number)
		self.projection_number_field.grid(row=1,column=6)
		
		#Called whenever variable projection_number is updated
		def Follow_variable(value):
			try:
				self.projection_number.set(str(int(self.projection_number.get())%len(self.angles)))
			except ValueError as err:
				template = "An exception of type {0} occurred. Arguments:\n{1!r}"
				message = template.format(type(err).__name__, err.args)
				eprint(message)

				Returnwindow = Toplevel()
				Returnwindow.geometry("")
				self.Returnwindow=Returnwindow
				self.Returnwindow.title('Invalid projection number: ')		
				self.button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy,width=50)
				self.button.pack()	
			else:
				self.watch()		
	
		self.projection_number_field.bind("<Return>", Follow_variable)
		
		self.viewplus=Button(frame,text='+', command=self.increase_projectionnumber)
		self.viewplus.grid(row=1, column=7)
		
		self.Nextbutton=Button(master,text='+',command=self.create_new_infoline)
		self.Nextbutton.grid()
		
		self.deletebutton=Button(frame,text='X',command=self.delete)
		self.deletebutton.grid(row=1,column=0)
		self.data=None
		self.angles=[0]
	
	### Increases the projection_number and calls watch to update the plot
	def increase_projectionnumber(self):
		try:
			self.projection_number.set(str(int(self.projection_number.get())+1))
			self.projection_number.set(str(int(self.projection_number.get())%len(self.angles)))
		except ValueError:
			self.projection_number.set(str(0))
		else:
			self.watch()		
		
	### Decreases the projection_number and calls watch to update the plot	
	def decrease_projectionnumber(self):
		try:
			self.projection_number.set(str(int(self.projection_number.get())-1))
			self.projection_number.set(str(int(self.projection_number.get())%len(self.angles)))
		except ValueError:
			self.projection_number.set(str(0))
		else:
			self.watch()		
		
	### Creates (or updates) a plot showing a projection of the corresponding data	
	def watch(self):
		#Load data only if the name changed
		if self.name_old!=self.entryfield.get():
			try:
				self.data=None
				self.angles=[0]
				self.name_old=''
				
				name=self.entryfield.get()
				if not os.path.isfile(replace_ext(name, 'rawtlt')) and not os.path.isfile(replace_ext(name, 'csv')):
					anglerequest=create_anglefile(name, '<channel>')
					anglerequest.ask()
				[data,angles]=h5.readh5(name)
				
				self.data=data
				self.angles=angles
				self.projection_number.set(str(0))
				self.name_old=self.entryfield.get()
			except IOError as err:
				template = "An exception of type {0} occurred. Arguments:\n{1!r}"
				message = template.format(type(err).__name__, err.args)
				eprint(message)
				Returnwindow = Toplevel()
				Returnwindow.geometry("")
				self.Returnwindow=Returnwindow
				self.Returnwindow.title('Unable to read data: ')		
				self.button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy,width=50)
				self.button.pack()	
				
		#Plot data if possible	
		if self.data is not None:
			try:
				if Cut_edges_choice.var.get()==1:
					cutval=float(Cut_edges_choice.entryfield.get())
				else:
					cutval=0.
				if Brightness_correction_choice.var.get()==1:	
					[data,angles,a]=h5.Preprocessing_steps_with_Brightness(self.data,self.angles,'',cutval,[])
				else:
					[data,angles,a]=h5.Preprocessing_steps_without_Brightness(self.data,self.angles,'',cutval,[])
							
				Na,Nx,Ny=data.shape
				data=data-np.percentile(data,vis_lower_percentile)
				data=data/np.percentile(data,vis_upper_percentile)
				thres=Thresholding_choice.var.get()
				if thres==1:
					data=np.array(data)
					threshold=float(Thresholding_choice.entryfield.get())
					data[np.where(data<threshold)]=0
									
				
				image=data[int(self.projection_number.get())%len(angles)]*255
				
				#Check wether a Plot must be created or only updated
				if not fignum_exists(self.figure_number):
					if self.Channelname.get()!='':
						self.fig=figure(self.Channelname.get())
					else:
						self.fig=figure(self.entryfield.get())
					self.figure_number=self.fig.number
					title(self.entryfield.get()+ ' projection number '+ str(int(self.projection_number.get())%len(angles))+' ('+str(round(180/np.pi*self.angles[int(self.projection_number.get())],2))+u'\N{DEGREE SIGN})'  )
						
					self.disp_img=imshow(image, cmap=cm.gray)
					self.disp_img.set_clim([0, 255*np.percentile(data,vis_upper_percentile)])
					show()
				else:
					figure(self.figure_number)
					if self.Channelname.get()!='':
						self.fig.canvas.set_window_title(self.Channelname.get())
						
					else:
						self.fig.canvas.set_window_title(self.entryfield.get())
					
					title(self.entryfield.get()+ ' projection number '+ str(int(self.projection_number.get())%len(angles) )+' ('+str(round(180/np.pi*self.angles[int(self.projection_number.get())],2))+u'\N{DEGREE SIGN})')
					self.disp_img.set_data(image)
					self.disp_img.set_clim([0, 255*np.percentile(data,vis_upper_percentile)])
					draw()
					cfm=get_current_fig_manager()
					cfm.window.attributes('-topmost', True)
					cfm.window.attributes('-topmost', False)
			
			#Error handling
			except Exception as err:
				template = "An exception of type {0} occurred. Arguments:\n{1!r}"
				message = template.format(type(err).__name__, err.args)
				eprint(message)
				Returnwindow = Toplevel()
				Returnwindow.geometry("")
				self.Returnwindow=Returnwindow
				self.Returnwindow.title('Unable to show data: ')
				
				self.button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy,width=50)
				self.button.pack()		

				
	### Allows for browser window to select data	
	def browspath(self):
		try:
			global startingpath
			fenster.filename = tkFileDialog.askopenfilename(initialdir=startingpath,title = "Select file",filetypes = (("MRC files","*.mrc"),("HDF5 files","*.h5"),("all files","*.*")))
			
			
			if fenster.filename!='':
				startingpath= '/'.join( fenster.filename.split('/')[:-1] )
				self.entryfield.delete(0, END)
				self.entryfield.insert(0,os.path.relpath(fenster.filename))
		except Exception as err:
			template = "An exception of type {0} occurred. Arguments:\n{1!r}"
			message = template.format(type(err).__name__, err.args)
			eprint(message)
			pass
	
	### Add a new line to the list of input lines 		
	def create_new_infoline(self):		
		C=self.Nextbutton
		C=C.grid_remove()
		Infilelist.append(Inputline(Infileframe.interior))
		
	### Delete an input line	
	def delete(self):
		if len(Infilelist)>1:
			if self==Infilelist[0]:
				Label(Infilelist[1].frame,text='Infile:').grid(row=0,column=1,sticky='W')
				Label(Infilelist[1].frame,text='Channel name:').grid(row=0,column=3,sticky='W')
				Label(Infilelist[1].frame,text='Weight:').grid(row=0,column=4,sticky='W')
				Label(Infilelist[1].frame,text='View:').grid(row=0,column=5,columnspan=3)
			Infilelist.remove(self)
			self.frame.grid_remove()
			close(self.figure_number)
			del self			
		else:
			close(self.figure_number)
			self.entryfield.delete(0, END)
			self.Channelname.delete(0,END)
			self.weight.delete(0,END)
			self.weight.insert(0,'0.01')
			self.projection_number.set('0')
			self.name_old=''
			self.figure_number=-1
			self.angles=[1]
			
### Widget Class allowing for parameter to be entered (e.g. Regularization parameters)
class Parameterchoice:
	def __init__(self,master,position,title,defaultvalue):
		self.frame=Frame(master)
		self.frame.grid(row=position,column=3,sticky='E')
		
		self.label=Label(self.frame,text=title)
		self.label.grid(row=position,column=3,sticky='E')
		self.entryfield=Entry(self.frame,width=20)
		self.entryfield.grid(row=position,column=4,sticky='E')
		self.entryfield.insert(0,defaultvalue)
		
		
def setMaxWidth(stringList, element):
    f = windowFont #tkFont.nametofont("TkDefaultFont")
    zerowidth=f.measure("0")
    w=int(max([f.measure(i) for i in stringList])/zerowidth*1.2)
    element.config(width=min(w,100))
		
### Widget Class with a Button and a textfield to enter parameters
class Optionbox:
	def __init__(self,master,List,position,title,initial=0):
		
		self.frame=Frame(master)
		self.frame.grid(row=position,column=1,sticky='W')
		self.label=Label(self.frame,text=title)
		self.label.grid(row=position,column=0,sticky='E')
		self.var = StringVar(self.frame)
		self.var.set(List[initial]) # initial value
		self.option = OptionMenu(self.frame,  self.var,*List)
		self.option.grid(row=position,column=1, sticky='W')
		setMaxWidth(List, self.option)
	

class create_anglefile():
	def __init__(self,name,channelname):
		self.name=name
		self.channelname=channelname
		
		self.Returnwindow = Toplevel()
		self.Returnwindow.title("No angle file for "+str(channelname))
		self.frame=Frame(self.Returnwindow)
		
		self.frame.grid()
		self.Label=Label(self.frame,text='Create angle file for '+str(channelname))
		self.Label.grid(column=0,row=0,columnspan=2) 
		self.Label=Label(self.frame,text='Use regularly spaced angles')
		self.Label.grid(column=0,row=1,columnspan=2) 

		self.entry1=Entry(self.frame)
		
		self.entry1.grid(column=1,row=2)
		self.Label1=Label(self.frame,text='Start angle: ')
		self.Label1.grid(column=0,row=2)
		self.entry2=Entry(self.frame)
		self.entry2.grid(column=1,row=3)
		self.Label2=Label(self.frame,text='Final angle: ')
		self.Label2.grid(column=0,row=3)
		self.entry3=Entry(self.frame)
		self.entry3.grid(column=1,row=4)
		self.Label3=Label(self.frame,text='Step size: ')
		self.Label3.grid(column=0,row=4)
		self.Button=Button(self.frame,text='Create File',command=self.create)
		self.Button.grid(column=0,columnspan=2,row=5)

		self.Label4 = Label(self.frame,text='')
		self.Label4.grid(column=0, columnspan=2, row=6)
		self.Button2 = Button(self.frame, text='Dismiss', command=self.Returnwindow.destroy, width=30)
		self.Button2.grid(column=0,columnspan=2,row=7)
		
		tip = ListboxToolTip(self.frame, ["The file "+name+ ' for '+channelname+' does not possess an angle file. This tool allows to create such a file for regularly spaced angles. With the initial and final angle, and the step size specified, the "Create file" button creates a corresponding rawtlt file. (No consistency checks with respect to the MRC file are performed.)' ],[450,90]) 

	def ask(self):
		fenster.wait_window(self.Returnwindow)
        
	def create(self):
		try:
			initial = float(self.entry1.get())
			final = float(self.entry2.get())
			step = max(1e-6, float(self.entry3.get()))
			filename= replace_ext(self.name, 'rawtlt')
			stream=open(filename,'w')

			i = initial
			while i <= final:
				stream.writelines(str(i)+'\n')
				i += step
				
			stream.flush()
			stream.close()
		except:
			message=tkMessageBox.showinfo('Error','Unable to create angle file for '+str(self.channelname)+'.\n Make sure your input is feasible.')
		else:
			self.Returnwindow.destroy()


### Function that extracts all the relevant information from the GUI and creates a corresponding commandline, or a list of complaints in case some information is not suitable. 
def Extract_Information():
	Complaint=[]
	
		
	Names=[]
	count=0
	for item in Infilelist:
		count+=1
		name=item.entryfield.get()
		if  not os.path.isfile( name ):
			Complaint.append('Input file number '+str(count)+': The file '+name+' was not found.')
		Names.append(item.entryfield.get())
			
	Channelnames=[]
	count=0
	for item in Infilelist:
		if item.Channelname.get()=='': #Give names to unnamed channels
			item.Channelname.insert(0,'Channel_'+str(count))
		count+=1
	count=0
	
	for item in Infilelist:
		count+=1
		if item.Channelname.get()=='':
			Complaint.append('Input file number '+str( count)+' does not possess a channel name.')
		Channelnames.append(item.Channelname.get())
	if len(Channelnames)!= len(set(Channelnames)):
		Complaint.append('The channel names are not unique.')

	count=0
	for item in Infilelist:
		count+=1
		name=item.entryfield.get()
		if name[len(name)-4:len(name)]=='.mrc':
			if not os.path.isfile(replace_ext(name, 'rawtlt')) and not os.path.isfile(replace_ext(name, 'csv')):
				anglerequest=create_anglefile(name,Channelnames[count-1])
				anglerequest.ask()
			if not os.path.isfile(replace_ext(name, 'rawtlt')) and not os.path.isfile(replace_ext(name, 'csv')):
				Complaint.append( 'Input file number '+str(count)+': No angle file was found.')


	
			
	Discrepancymethod=Discrepancychoice.var.get()
	Regulartionmethod=Regularisationchoice.var.get()
	Coupling=Couplingchoice.var.get()	
	
	try:
		alpha=Alphachoice.entryfield.get() 
		if Regulartionmethod=='TGV':
			alpha=[float(alpha),1.]
		elif Regulartionmethod=='TV' :
			alpha=[1.]
	except ValueError:
		Complaint.append('Could not interpret the ratio for regularization, please enter an integer or float value.')
	
	try:
		float(Total_Reg_parameter.entryfield.get())
	except ValueError:
		Complaint.append('Could not interpret global regularization parameter.')
	else:
		mu=[]
		count=0
		for item in Infilelist:
			try:
				mu_item=float(item.weight.get())
				mu.append(mu_item*float(Total_Reg_parameter.entryfield.get()))
			except ValueError:	
				Complaint.append('Could not interpret weight in input file line '+str(count)+', please enter an integer or float value.')
			count+=1	

	try:
		maxiter=int(Maxiterchoice.entryfield.get())
	except ValueError:
		Complaint.append('The value of the number of iterations could not be interpreted, please enter an integer, e.g. "1000".') 
	bright=Brightness_correction_choice.var.get()
	thres=Thresholding_choice.var.get()
	if thres==1:
		try:
			thres=float(Thresholding_choice.entryfield.get())
		except ValueError:
			Complaint.append('Could not interpret background threshold value, please enter a number between 0 and 1, e.g. "0.05".')
	cut=Cut_edges_choice.var.get()
	if cut==1:
		try:
			cut=float(Cut_edges_choice.entryfield.get())	
		except ValueError:
			Complaint.append('Could not interpret crop detector boundary value, please enter a number between 0 and 1, e.g. "0.1".')
		
	Bad=Badprojection_choice.var.get()
	Badsearch=[0, 0]
	Bad_file=''
	Bad_direct=[]
	
	if Bad==1:
		if Badprojection_choice.Firstbutton.var.get()==1:
			A=Badprojection_choice.Firstbutton.entryfield.get()
			A=A.replace(' ',',')
			A=A.split(',')
			while '' in A:
				A.remove('')
			try:
				for a in A:
					Bad_direct.append(int(a))
			except ValueError:
				Complaint.append('Could not interpret values for directly entered bad projections.')  
		if Badprojection_choice.Secondbutton.var.get()==1:
			Bad_file=Badprojection_choice.Secondbutton.entryfield.get()
			if  not os.path.isfile( Bad_file ):
				Complaint.append('Bad projection text file does not exist.')
		#if Badprojection_choice.Thirdbutton.var.get()==1:
		#	try:
		#		Badsearch=[float(Badprojection_choice.Thirdbutton.entryfield1.get()),float(Badprojection_choice.Thirdbutton.entryfield2.get())]
		#	except ValueError:
		#		Complaint.append('Parameter for the  search of bad projections could not be understood')
			
	slices=[]
	try:
		slices.append(int(slices_choice.start.get()))
	except ValueError:
		if slices_choice.start.get()!='':
			Complaint.append('Could not interpret the start value in slices to process.')
		else:
			slices_choice.start.insert(0,'0')
			slices.append(int(slices_choice.start.get()))
	try:
		slices.append(int(slices_choice.end.get()))
	except ValueError:	
		if slices_choice.end.get()!='':
			Complaint.append('Could not interpret the end value in slices to process.')	
		else:
			try:
				name=Names[0]
				[data,angles]=h5.readh5(name)
				Nz=data.shape[1]-1
				slices_choice.end.insert(0,str(Nz))
				slices.append(int(slices_choice.end.get()))	
			except:
				slices_choice.end.insert(0,'0')
				slices.append(int(slices_choice.end.get()))	
				Complaint.append('Could not interpret slice information.')	
				Complaint.append('Unable to read data for '+ Channelnames[0])
	try:
		slices.append(int(slices_choice.step.get()))
	except ValueError:
		Complaint.append('Could not interpret the step value in slices to process.')	
	
	plot=[]
	try:
		plot.append(Plotbutton.var.get())		
		if  plot[0]==1:
			plot.append(float(Plotbutton.entryfield.get()))
			if plot[1]!=math.floor(plot[1]):
				if plot[1]>1:
					Complaint.append('Plot frequency must be an integer or float between 0 and 1, e.g. "100" or "0.1".')
		else: plot.append(float(Plotbutton.entryfield.get()))
	except:
		Complaint.append('Could not interpret the plot frequency value.')

	overlapping=[]
	try:
		overlapping.append(Overlappingbutton.var.get())
		if overlapping[0]==1:
			overlapping.append(int(Overlappingbutton.entryfield.get()))
		else:
			overlapping.append(0)
	except:
		Complaint.append('Could not interpret overlapping slices value.') 

	my_gpu_devices,GPUchoice.var.get()
	count=0

	while str(my_gpu_devices[count])!=GPUchoice.var.get():
		count+=1
	GPU=count
	
	commandline=''
	
	
	#Complaining about input from the GUI
	if len(Complaint)>0:
		Text='Error, data could not suitably be interpreted.'
		count=0
		for con in Complaint:
			count+=1
			Text+= '\n'+str(count)+ '): '+con
		message=tkMessageBox.showinfo('Complaints',Text)
	else:
		commandline=str(sys.executable)
		commandline+=' Reconstruction_coupled.py '
		for name in Names:
			commandline+='"'+name+'"'+' '
		commandline+='--Outfile '+'"<outputfile>" '#'"'+Out+'" '
		
		commandline+='--alpha '
		for a in alpha:
			commandline+=str(a)+' '
		commandline+='--mu '
		for a in mu:
			commandline+=str(a)+' '
		commandline+='--Maxiter '+str(maxiter)+' '
		
		commandline+='--Regularisation '+Regulartionmethod+' '
		commandline+='--Discrepancy '+Discrepancymethod+' '
		

		count=0
		while Couplingoptions_long[count]!=Coupling:
			count+=1
		
		commandline+='--Coupling '+Couplingoptions_short[count]+' '
		commandline+='--SliceLevels '
		for i in range(0,3):
			commandline+=str(slices[i])+' '
		
		commandline+='--Channelnames '
		for name in Channelnames:
			commandline+='"'+name+'" '
			
		commandline+='--Datahandling '
		if bright==1:
			commandline+='bright '
		else:
			commandline+='basic '
		
		if thres!=0:
			commandline+='thresholding '
		else:
			commandline+='"" '
		
		commandline+=str(cut)+' '
		commandline+='"'+Bad_file+'" '
		
		commandline+=str(thres)+' '
		
		if len(Bad_direct)>0:
			commandline+='--Badprojections '
			for a in Bad_direct:
				commandline+=str(a)+' '
				
		commandline+='--Plot '+str(plot[0])+' '+str(plot[1])+' '
		
		commandline+='--GPU_Choice '+str(GPU)+' '
		
		commandline+='--Find_Bad_Projections '+str(Badsearch[0])+' '+str(Badsearch[1])+' '
		
		commandline+='--Overlapping '+str(overlapping[1])+' '
	
	
	
	
	if len(Complaint)==0:
		eprint( 'Command:', commandline)
	else:
		eprint ('Complaints:', Complaint)
	return commandline, Complaint

### Function calls Extract_Information to gather information, and then display the corresponding commandline or a list of complaints concerning the input	
def Extract_Information_show():
	commandline,complaint=Extract_Information()
	commandline=commandline.replace('<outputfile>',Extract_Information_outfield.entryfield.get())
	if len(complaint)==0:
		Returnwindow = Toplevel()
		Returnwindow.title("Command line")
		
		button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy)
		button.pack()	
		
		ErrorLabel=Label(Returnwindow,text='Can be executed via:')
		ErrorLabel.pack()
		scrollbar = Scrollbar(Returnwindow)
		scrollbar.pack(side=RIGHT, fill=Y)
		listbox = Text(Returnwindow, yscrollcommand=scrollbar.set)
		listbox.insert(END,'cd '+os.getcwd()+'\n')
		listbox.insert(END,commandline)
		listbox.pack(side=LEFT, fill=BOTH)
		listbox.bind("<1>", lambda event: listbox.focus_set())
		listbox.config(state=DISABLED)
		scrollbar.config(command=listbox.yview)




#################################
class AsynchronousFileReader(threading.Thread):
    '''
    Helper class to implement asynchronous reading of a file
    in a separate thread. Pushes read lines on a queue to
    be consumed in another thread.
    '''

    def __init__(self, fd, my_queue):
        assert isinstance(my_queue, queue.Queue)
        assert callable(fd.readline)
        threading.Thread.__init__(self)
        self._fd = fd
        self._queue = my_queue

    def run(self):
        '''The body of the tread: read lines and put them on the queue.'''
        for line in iter(self._fd.readline, ''):
            self._queue.put(line)
            

    def eof(self):
        '''Check whether there is no more content to expect.'''
        return not self.is_alive() and self._queue.empty()
    
### Class derived from Thread which runs an command as subprocess and reads to stderr and stdout stream from the subprocess and saves them 
class ThreadedTask2(threading.Thread):
	def __init__(self, queue,command,Computationfield):
		self.command=command
		self.queue = queue
		self.Computationfield=Computationfield
		self._sleepperiod=1
		self._stopevent = threading.Event( )
		threading.Thread.__init__(self)
		Computationfield.information=command+'\n'
		self.Computationfield.whole_story=command+'\n'
		Computationfield.errorinformation=''
	
	def joining(self):
		""" Stop the thread and wait for it to end. """
		self._stopevent.set( )
		#threading.Thread.join(self, None)
		print('############ Thread2 Closed #############')

	
	def run(self):
		'''
		Example of how to consume standard output and standard error of
		a subprocess asynchronously without risk on deadlocking.
		'''

		# Launch the command as subprocess.
		self.process = subprocess.Popen(shlex.split(self.command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		# create queues
		def enqueue_output(out, queue):
			for line in iter(out.readline, b''):
				queue.put(line)
			queue.put(None)
			out.close()
		
		stdout = queue.Queue()
		stdout_thread = threading.Thread(target=enqueue_output,
							   args=(self.process.stdout, stdout))
		stdout_thread.start()
		
		stderr = queue.Queue()
		stderr_thread = threading.Thread(target=enqueue_output,
							   args=(self.process.stderr, stderr))
		stderr_thread.start()

		# put output in window
		noexit = True
		while (stdout is not None) or (stderr is not None):
			if self.process.poll() is not None:
				self.process.wait()
				noexit = False
			if self._stopevent.isSet():
				self.process.terminate()
				noexit = False
				
			if not noexit:
				# Let's be tidy and join the threads we've started.
				stdout_thread.join()
				stderr_thread.join()
				
			if stdout is not None:
				if not stdout.empty():
					line = stdout.get()
					if line is None:
						stdout = None
					else:
						line = line.decode()
						self.Computationfield.whole_story+=str(line)
						self.Computationfield.information+=str(line)

			if stderr is not None:
				if not stderr.empty():
					line = stderr.get()
					if line is None:
						stderr = None
					else:
						line = line.decode()
						self.Computationfield.errorinformation+=str(line)
						self.Computationfield.whole_story+=str(line)

			keyword='Overall progress at '
			keywordposition=self.Computationfield.information.rfind(keyword)+len(keyword)
			if keywordposition!=-1:
				percentage=self.Computationfield.information[keywordposition:keywordposition+3]
				percentage_old=''
				if percentage!=percentage_old:
					percentage_old=percentage
					percentage=percentage.replace(' ','')
					percentage=percentage.replace('%','')
					self.Computationfield.Progressbutton.configure(text='Reconstruction at ' +str(percentage)+'% for: ')
					percentage_old=percentage
						
			# Sleep a bit before asking the readers again.	
			time.sleep(.1)
			
		# Case in which the tasks are joined before terminating
		if self._stopevent.isSet():
			#print ('Aborted')
			self.Computationfield.Progressbutton.configure(text='Reconstruction aborted for: ')
			self.Computationfield.success=-1
		#Case the subprocess terminates successful
		elif self.process.poll( )==0:
			#print( 'Successfully terminated')
			self.Computationfield.Progressbutton.configure(text='Reconstruction complete for: ')
			self.Computationfield.success=1
			self.Computationfield.stoppingbut.stoppingbutton.configure(text='Delete',command=self.Computationfield.stoppingbut.destroyline)
			self.Computationfield.savebutton=savebutton(self.Computationfield)
			
		#Case the subprocess terminates due to an error
		else:
			#print('An error occured')			
			self.Computationfield.Progressbutton.configure(text='Critical error occurred for: ')
			self.Computationfield.success=-2
			self.Computationfield.stoppingbut.stoppingbutton.configure(text='Delete',command=self.Computationfield.stoppingbut.destroyline)			
		
		## # Close subprocess' file descriptors.
		## self.process.stdout.close()
		## self.process.stderr.close()
		## ##########		

### Class derived from Thread which creates a thread waiting for self._stopevent to be set ( in which case a subthread is joined) or after said subthread is terminated.	
class ThreadedTask(threading.Thread):
	def __init__(self, queue,command,Computationfield):
		self.command=command
		self.queue = queue
		self.Computationfield=Computationfield
		self._sleepperiod=1
		self._stopevent = threading.Event( )
		threading.Thread.__init__(self)
		self.information=command
		self.errorinformation=''
	def joining(self):
		""" Stop the thread and wait for it to end. """
		self._stopevent.set( )
		threading.Thread.join(self, None)
		self.task.joining()
		print('############ Thread Closed #############')

	def run(self):
		""" main control loop """
		print ("%s starts" % (self.getName( ),))
		count = 0
			
		self.task=ThreadedTask2(self.queue,self.command,self.Computationfield)
		self.task.start()
		
		while not self._stopevent.isSet( ) and self.task.is_alive():		
			#print 'self._stopevent.isSet( )',self._stopevent.isSet( )
			count+=1
			#print ("loop %d" % (count,))
			self._stopevent.wait(self._sleepperiod)
		 
	

### Widget Class depicting the field with options to start and control  computations and react to the current status of the computation		
class computationbutton():
	Computationcounter=0
	def __init__(self,predecessor=None,frame=fenster):
		self.predecessor=predecessor
		self.successor=None
		self.master = frame
		self.computationbuttonframeBig=frame
		self.computationbuttonframe=Frame(self.computationbuttonframeBig)
		self.computationbuttonframe.pack(side=TOP)
		self.show_window=None
		
		self.Progressbutton=Label(self.computationbuttonframe,text='Reconstruction: ')
		self.Progressbutton.grid(row=0,column=0)
		
		self.name= StringVar()
		self.namefield=Entry(self.computationbuttonframe,textvariable=self.name)
		self.namefield.insert(END,'Computation'+str(computationbutton.Computationcounter))
		computationbutton.Computationcounter+=1
		self.namefield.grid(row=0,column=1)
		
		self.Computationbutton=Button(self.computationbuttonframe,text='Start reconstruction',command=self.computation)
		self.Computationbutton.grid(row=0,column=2)
		
		
		
		self.whole_story=''
		self.queue = queue.Queue()
		self.Complaint=[1]
		
		self.number=int(np.random.random()*10000000)
		
		self.success=None


	# Function is called when a computation is started	
	def computation(self):
				
		if len(self.Complaint)>0:
			commandline,Complaint=Extract_Information()
			
			self.commandline=commandline
			self.Complaint=Complaint
			self.name=self.namefield.get()
			
		if len(self.Complaint)==0:
			self.success=0
			
			Channelnames=[]
			count=0
			for item in Infilelist:
				Channelnames.append(item.Channelname.get())
			if len(Infilelist)>1:
				Channelnames.append('Collection')
				Channelnames.append('Collection2')
			self.Channelnames=Channelnames
			#self.Outputadress=Outputaddress.entryfield.get()
			
			slices=[]
			slices.append(int(slices_choice.start.get()))
			slices.append(int(slices_choice.end.get()))
			slices.append(int(slices_choice.step.get()))
			self.slices=range(slices[0],slices[1]+1,slices[2])
			
			self.commandline=self.commandline.replace('"<outputfile>"','tmp/'+str(self.number)+'/result')
			
			self.task=ThreadedTask(self.queue,self.commandline,self)
	
			self.task.start()
			self.commandline=self.commandline.replace('tmp/'+str(self.number)+'/result','"<outputfile>" ')
			
			self.Computationbutton.grid_forget()
			self.Progressbutton=Label(self.computationbuttonframe,text='Computation is in progress for:')
			self.Progressbutton.grid(row=0,column=0)
			self.Detailbutton=Button(self.computationbuttonframe,text='Details',command=self.Show_details)
			self.Detailbutton.grid(row=0,column=3)
			self.stoppingbut=stoppingbutton(self.task,self.computationbuttonframe,self)
			output=0
			if self.successor==None:
				self.successor=computationbutton(self,self.master)
			self.Check_wether_finished()
			self.Show_details()
			return output
	
	#Repeatedly checks wether the task is finished
	def Check_wether_finished(self):
		if self.task.is_alive():
			fenster.after(1000, self.Check_wether_finished) 
		else:
			self.Show_details
		
	#Create an Window which shows information concerning the reconstruction		
	def Show_details(self,force=False):
		if self.show_window is None:
			self.show_window=Show_details_window(self)
		elif not self.show_window.Returnwindow.winfo_exists()or force==True:
			self.show_window=Show_details_window(self)
		elif force==False:
			self.show_window.update()
		self.show_window.Returnwindow.lift()

			

		
### Widget Class. A new window showing as title the status of the reconstruction, and depending on it showing a viewer allowing to consider reconstuctions		
class Show_details_window():
	def __init__(self,computation):
		self.count=0
		self.computation=computation
		self.success=computation.success
		Returnwindow = Toplevel()
		
		self.Returnwindow=Returnwindow
		self.Stated_aboard=0
				
		if computation.success ==1:
			self.Returnwindow.title('Successful reconstruction  of: '+computation.namefield.get())
		elif computation.success==-1:
			self.Returnwindow.title('Aborted reconstruction  of: '+computation.namefield.get())
		elif computation.success==-2:
			self.Returnwindow.title('Critical error in reconstruction: '+computation.namefield.get())
			
		#self.button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy)
		#self.button.pack()		
		
		
		
		if computation.success==1:
			
			
			Bigframe=Frame(self.Returnwindow,borderwidth=4, pady=10)
			Bigframe.pack(fill=X)
			
			Titleframe=Frame(Bigframe,borderwidth=4,relief="groove")
			Titleframe.pack(anchor='w',fill=X)
			
			Title=Label(Titleframe,text="Individual results", font=headingFont)
			Title.pack(anchor='w')

			
						
			command=computation.commandline
			self.showimage_frame=VerticalScrolledFrame(Bigframe,min(len(range(len(computation.Channelnames)))*40,150),borderwidth=4,relief="sunken",width=10000,height=100000)
			self.showimage_frame.pack(anchor='w',fill=X)	
			self.plotsolution_list=[]
			for i in range(len(computation.Channelnames)):
				self.plotsolution_list.append(showimages(computation,'tmp/'+str(computation.number)+'/result',computation.Channelnames[i],computation.slices,self.showimage_frame.interior))
			if len(computation.Channelnames)>1:
				self.plotsolution_list[len(computation.Channelnames)-2].label.config(text='All channels in original ratio')
				self.plotsolution_list[len(computation.Channelnames)-1].label.config(text='All channels in normalized ratio')
		

		
		Bigframe=Frame(self.Returnwindow,borderwidth=4)
		Bigframe.pack(fill=BOTH)
		
		Titleframe=Frame(Bigframe,borderwidth=4,relief="groove")
		Titleframe.pack(anchor='w',fill=X)
		
		Title=Label(Titleframe,text="Computational details", font=headingFont)
		Title.pack(anchor='w')
		self.scrollbar = Scrollbar(Returnwindow)
		
		self.listbox = Text(Returnwindow, yscrollcommand=self.scrollbar.set)
		self.listbox.insert(END,computation.whole_story)
		self.known=computation.whole_story
		self.listbox.pack(side=LEFT, fill=BOTH,anchor='w')
		self.scrollbar.pack(side=RIGHT, fill=Y,anchor='w')
		self.listbox.config(state=DISABLED)
		self.listbox.bind("<1>", lambda event: self.listbox.focus_set())
		self.scrollbar.config(command=self.listbox.yview)
		self.update()
		
	#update the Window showing details of reconstruction process
	def update(self):
		if self.computation.success==0:
			self.Returnwindow.title('Reconstruction in process for: '+self.computation.namefield.get())
		if self.computation.success==1:
			self.Returnwindow.title('Successful reconstruction details of: '+self.computation.namefield.get())
		elif self.computation.success==-1:
			self.Returnwindow.title('Aborted reconstruction of: '+self.computation.namefield.get())
		elif self.computation.success==-2:
			self.Returnwindow.title('Critical error in reconstruction: '+self.computation.namefield.get())
		
				
		if (self.computation.success==1 or self.computation.success==-2) and self.success==0:
			self.listbox.config(state=NORMAL)
			if self.count>1:
				self.listbox.delete('0.0',END)		
			self.listbox.insert(END,"Finished")
			self.computation.Show_details(True)
			self.Returnwindow.destroy()
			return
		if self.computation.success==-1 and self.Stated_aboard==0:
			self.listbox.config(state=NORMAL)	
			self.listbox.insert(END,"\n Code was manually aborted!")
			self.listbox.config(state=DISABLED)
			self.Stated_aboard=1
		self.count+=1
		
		if self.computation.success==0:	
			storylength=len(self.known)
			newstorylength=len(self.computation.whole_story)
			if self.known!=self.computation.whole_story:
				self.listbox.config(state=NORMAL)
				self.listbox.insert(END,self.computation.whole_story[storylength:newstorylength])
				self.listbox.config(state=DISABLED)
				self.known=self.computation.whole_story[0:newstorylength]
				
		if not ((self.computation.success==1 or self.computation.success==-2)and self.success==0): 
			self.Returnwindow.after(1000, self.update) # run itself again after 1000 ms
	
	
	
	

	
	
### Widget class allowing to plot a sequence of images and step through the slices 
class showimages():
	def __init__(self,computation,source,channelname,slices,Returnwindow):
		self.Returnwindow=Returnwindow
		self.frame=Frame(Returnwindow)
		self.frame.pack(fill=X,anchor='w')
		self.source=source
		self.computation=computation
		self.slices=slices
		self.channelname=channelname
		self.label=Label(self.frame,text=channelname,width=40,anchor='w')
		self.label.grid(row=0,column=0,sticky='W')
		self.data=mrcfile.open(source+channelname+'.mrc').data
		self.Nz=self.data.shape[0]
		
		self.viewminus=Button(self.frame,text='-', command=self.decrease_projectionnumber)
		self.viewminus.grid(row=0, column=1)
		
		self.figure_number=-1
		self.projection_number=StringVar()
		self.projection_number.set('0')
		self.projection_number_field=Entry(self.frame,width=4,text=self.projection_number)
		self.projection_number_field.grid(row=0,column=2)
		#activated when self.projection_number is updated to call watch if input is suitable 
		def Follow_variable_solution(value):
			try:
				self.projection_number.set(str(int(self.projection_number.get())%self.Nz))
			except ValueError as err:
				template = "An exception of type {0} occurred. Arguments:\n{1!r}"
				message = template.format(type(err).__name__, err.args)
				eprint(message)

				Returnwindow = Toplevel()
				Returnwindow.geometry("")
				self.Returnwindow=Returnwindow
				self.Returnwindow.title('Invalid slice number: ')		
				self.button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy,width=50)
				self.button.pack()	
			else:
				self.watch()

		self.projection_number_field.bind("<Return>", Follow_variable_solution)
		
		self.viewplus=Button(self.frame,text='+', command=self.increase_projectionnumber)
		self.viewplus.grid(row=0, column=3)

		self.button=Button(self.frame,text='Save',command=self.save)
		self.button.grid(row=0,column=4)
	def save(self):
		try:
			global startingpath
			
			name = tkFileDialog.asksaveasfilename(initialdir=startingpath,title = "Save data for channel "+self.channelname+" in "+self.computation.namefield.get(),filetypes = (("MRC files","*.mrc"),("all files","*.*")),initialfile = self.computation.namefield.get()+self.channelname+'.mrc')						
			
			if name!='':
				startingpath= '/'.join( name.split('/')[:-1] )
				try:#Remove .mrc
					if name[len(name)-4:len(name)]=='.mrc':
						name=name[0:len(name)-4]
				except:
					pass
				shutil.copyfile(self.source+self.channelname+'.mrc',name+'.mrc') 
				if self.channelname not in ['Collection','Collection2']:
					shutil.copyfile(self.source+self.channelname+'sinogram.mrc',name+'sinogram.mrc') 
				
		except Exception as err:
			template = "An exception of type {0} occurred. Arguments:\n{1!r}"
			message = template.format(type(err).__name__, err.args)
			eprint(message)
			Returnwindow = Toplevel()
			Returnwindow.geometry("")
			self.Returnwindow=Returnwindow
			self.Returnwindow.title('An error during Saving: ')			
			self.button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy,width=50)
			self.button.pack()	


	# Increases the value of the slice we are currently at and plots data or updates plot
	def increase_projectionnumber(self):
		try:
			self.projection_number.set(str(int(self.projection_number.get())+1))
			self.projection_number.set(str(int(self.projection_number.get())%self.Nz))
		except ValueError:
			self.projection_number.set(str(0))
		else:
			self.watch()		
		
	# Decreases the value of the slice we are currently at and plots data or updates plot
	def decrease_projectionnumber(self):
		try:
			self.projection_number.set(str(int(self.projection_number.get())-1))
			self.projection_number.set(str(int(self.projection_number.get())%self.Nz))
		except ValueError:
			self.projection_number.set(str(0))
		else:
			self.watch()		
		
	#Plot or update plot concerning the solutions
	def watch(self):
		try:		
			data=np.array(self.data)									
			image=data[int(self.projection_number.get())%self.Nz]*255
			
			if not fignum_exists(self.figure_number):
				self.fig=figure(self.computation.namefield.get()+': '+self.channelname)	
				
				self.figure_number=self.fig.number
				title('Slice number '+ str(self.slices[int(self.projection_number.get())%self.Nz ]))	
				self.disp_img=imshow(image, cmap=cm.gray)
				self.disp_img.set_clim([0, 255*np.percentile(data,vis_upper_percentile)])
				show()
			else:
				figure(self.figure_number)
				
				self.fig.canvas.set_window_title(self.computation.namefield.get()+': '+self.channelname)
				title('Slice number '+ str(self.slices[int(self.projection_number.get())%self.Nz ]))
				self.disp_img.set_data(image)
				self.disp_img.set_clim([0, 255*np.percentile(data,vis_upper_percentile)])
				draw()
				cfm=get_current_fig_manager()
				cfm.window.attributes('-topmost', True)
				cfm.window.attributes('-topmost', False)


		except Exception as err:
			template = "An exception of type {0} occurred. Arguments:\n{1!r}"
			message = template.format(type(err).__name__, err.args)
			eprint(message)
			Returnwindow = Toplevel()
			Returnwindow.geometry("")
			self.Returnwindow=Returnwindow
			self.Returnwindow.title('Unable to show data: ')			
			self.button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy,width=50)
			self.button.pack()		

class savebutton():
	def __init__(self,computation):
		self.computation=computation
		self.button=Button(computation.computationbuttonframe,text='Save',command=self.save)
		self.button.grid(row=0,column=5)
	def save(self):
		try:
			global startingpath
			name = tkFileDialog.asksaveasfilename(initialdir=startingpath,title = "Save data for "+self.computation.namefield.get(),filetypes = (("all files","*.*"),("MRC files","*.mrc")),initialfile = self.computation.namefield.get())						
			
			if name!='':
				startingpath= '/'.join( name.split('/')[:-1] )
				if name[len(name)-4:len(name)]=='.mrc':
					name=name[0:len(name)-4]

				for channel in self.computation.Channelnames:
					shutil.copyfile('tmp/'+str(self.computation.number)+'/result'+channel+'.mrc',name+channel+'.mrc') 
					if channel not in ['Collection','Collection2']:
						shutil.copyfile('tmp/'+str(self.computation.number)+'/result'+channel+'sinogram.mrc',name+channel+'sinogram.mrc') 
				configname=glob.glob('tmp/'+str(self.computation.number)+'/result*.config')[0]
				shutil.copyfile(configname,name+'.config')
		except Exception as err:
			template = "An exception of type {0} occurred. Arguments:\n{1!r}"
			message = template.format(type(err).__name__, err.args)
			eprint(message)
			Returnwindow = Toplevel()
			Returnwindow.geometry("")
			self.Returnwindow=Returnwindow
			self.Returnwindow.title('An error occured during saving: ')
			self.button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy,width=50)
			self.button.pack()		



		
### WIdget Class used to stop a running computation		
class stoppingbutton():
	def __init__(self,task,master,ComputationButton):
		self.master = master
		self.task=task
		self.ComputationButton=ComputationButton
		self.stoppingbutton=Button(self.master,text='Abort',command=self.Stopp)
		self.stoppingbutton.grid(row=0,column=4)
	
	#Stop task currently running
	def Stopp(self):
		#self.task.terminate()
		self.task.joining()
		self.ComputationButton.Progressbutton.configure(text='Computation aborted for: ')
		self.ComputationButton.Computationbutton.configure(text='Restart',state='normal')
		self.ComputationButton.Computationbutton.grid(row=0,column=2)
		try:
			self.stoppingbutton.configure(text='Delete',command=self.destroyline)
		except:
			pass #Avoid error if button is not yet set
		try:
			number=self.ComputationButton.number
			shutil.rmtree('tmp/'+str(number))
		except:
			pass
	
	#Delete current computation
	def destroyline(self):
		if self.ComputationButton.predecessor!=None:
			self.ComputationButton.predecessor.successor=self.ComputationButton.successor
		else:
			initial_computationbutton=self.ComputationButton.successor
			
		if self.ComputationButton.successor!=None:
			self.ComputationButton.successor.predecessor=self.ComputationButton.predecessor
		else:
			self.ComputationButton.predecessor.successor=None
		
		self.ComputationButton.computationbuttonframe.pack_forget()
		number=self.ComputationButton.number
		try:
			shutil.rmtree('tmp/'+str(number))
		except:
			pass
		if 	self.ComputationButton.show_window.Returnwindow.winfo_exists()==True:
			self.ComputationButton.show_window.Returnwindow.destroy()
		del self.ComputationButton
		del self
		
		
#################################################################################################################################################################
#Coupling methods
#Couplingoptions_long=['2D uncoupled','2D with Frobenius norm','2D with nuclear norm','3D uncoupled', '3D with Frobenius norm','3D with nuclear norm']
#Couplingoptions_short=['UNCOR2D', 'FROB2D','NUCL2D','UNCOR3D','FROB3D','Nucl3d']
# Nucl3D deactivated (experimental implementation)
Couplingoptions_long=['2D uncoupled','2D with Frobenius norm','2D with nuclear norm','3D uncoupled', '3D with Frobenius norm']
Couplingoptions_short=['UNCOR2D', 'FROB2D','NUCL2D','UNCOR3D','FROB3D'] 




# http://tkinter.unpythonic.net/wiki/VerticalScrolledFrame
class VerticalScrolledFrame(Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling
    """
    def __init__(self, parent, size, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)            

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self, bd=0, highlightthickness=0,yscrollcommand=vscrollbar.set,height=size)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)




#Inputfile Frame		
Infileframeall=Frame(fenster,borderwidth=4)
Infileframeall.pack(side=LEFT,anchor='n',fill=BOTH)


Infileframebig=Frame(Infileframeall,borderwidth=4,relief="groove")
Infileframebig.pack(fill=X)
Infileframe=VerticalScrolledFrame(Infileframeall,10000,borderwidth=4,relief="sunken",width=10000,height=800)

Infileframe.pack(side=TOP,anchor='n',fill=X)

Infileframe_bottom=Frame(Infileframeall)
Infileframe_bottom.pack(side=BOTTOM)


Infileframetitle=Label(Infileframebig,text="Input files", font=headingFont)
Infileframetitle.pack()

A=Inputline(Infileframe.interior)
Label(A.frame,text='File name:').grid(row=0,column=1,sticky='W')
Label(A.frame,text='Channel name:').grid(row=0,column=3,sticky='W')
Label(A.frame,text='Weight:').grid(row=0,column=4,sticky='W')
Label(A.frame,text='View:').grid(row=0,column=5,columnspan=3)
Infilelist.append(A)


#Preprocessingframe
Preprocssingframeall=Frame(fenster,borderwidth=4,width=10000,height=100000, pady=10)
Preprocssingframeall.pack(fill=X)
Preprocssingframebig=Frame(Preprocssingframeall,borderwidth=4,relief="groove",width=10000,height=100000)
Preprocssingframebig.pack(fill=X)
Preprocssingframe= Frame(Preprocssingframeall,borderwidth=4,relief="sunken",width=10000,height=100000)
Preprocssingframe.pack(fill=X)
Preprocssingframtitle=Label(Preprocssingframebig,text="Preprocessing options", font=headingFont)
Preprocssingframtitle.pack()

pos=1
Brightness_correction_choice=Brightness(Preprocssingframe,pos)
tip = ListboxToolTip(Brightness_correction_choice.button, ["When activated, the program tries to eliminate variations in the intensity between projections from different angles."],[450,45]) 

Brightness_correction_choice.button.select()
pos+=1
Thresholding_choice=Button_with_fields(Preprocssingframe,pos,'Perform background thresholding   ','value: ',1,'0.05')
tip = ListboxToolTip(Thresholding_choice.frame, ["When activated, the program thresholds the values below the given value to zero (where the max value is 1 and min value 0). This can be used to remove static noise outside the sample and to find a suitable baseline."],[450,80])

pos+=1
Cut_edges_choice=Button_with_fields(Preprocssingframe,pos,'Crop detector boundaries                ','value: ',0,'0.01')
tip = ListboxToolTip(Cut_edges_choice.frame, ["When activated with value x, this fraction of the outer detector data are discarded (on each side). This can be useful if the data captures much more (blank) space than the sample."],[450,60])

pos+=1
Badprojection_choice=Badprojections(Preprocssingframe,pos,'Exclude bad projections',['direct list:','text file:            ','Search:'])
tip = ListboxToolTip(Badprojection_choice.button, ["When activated, allows to designate projections (numbered from 0 to number of projections) which are discarded for reconstruction (since they do not contain suitable information)."],[450,60])
tip = ListboxToolTip(Badprojection_choice.Firstbutton.button, ["Enter a list of projection numbers which are to be discarded, seperated by space or ','."],[450,45])
tip = ListboxToolTip(Badprojection_choice.Secondbutton.button, ["Enter the name of a text file containing the projections to be discarded, listed one per line."],[470,45])
#tip = ListboxToolTip(Badprojection_choice.Thirdbutton.button, ["Algorithm   tries to find the bad projections itself (can take some time) requires two parameters, first on the amount of projections to be discarded (0.1=10%) and search radius (typically 1/10 number of detectors)"],[450,60])



#Reconstructionframe
Reconstructionframeall=Frame(fenster,borderwidth=4,width=10000,height=100000, pady=10)
Reconstructionframeall.pack(fill=X)
Reconstructionframebig=Frame(Reconstructionframeall,borderwidth=4,relief="groove",width=10000,height=100000)
Reconstructionframebig.pack(fill=X)
Reconstructionframe= Frame(Reconstructionframeall,borderwidth=4,relief="sunken",width=10000,height=100000)
Reconstructionframe.pack(fill=X)
Reconstructionframetitle=Label(Reconstructionframebig,text="Regularization parameters", font=headingFont)
Reconstructionframetitle.pack()

pos=1
Regularisationchoice=Optionbox(Reconstructionframe,['TGV','TV'],pos,'Regularization: ',0)
tip = ListboxToolTip(Regularisationchoice.frame, ["Chooses which regularization type to use: TGV is better suited for continuous transitions, TV for piecewise constant reconstructions."],[450,45]) 

Alphachoice=Parameterchoice(Reconstructionframe,pos,'Regularization order ratio: ','4')
tip = ListboxToolTip(Alphachoice.frame, ["Chooses the ratio of first and second order regularization parameter in TGV regularization (irrelevant for TV). The fixed value 4 is usually suitable and does not need to be adapted."],[300,120]) 

pos +=1
Couplingchoice=Optionbox(Reconstructionframe,Couplingoptions_long,pos,'Coupling type:  ',4)
tip = ListboxToolTip(Couplingchoice.frame, ["Chooses which kind of coupling to use in the regularization, choosing between 2D and 3D reconstruction (couping in z dimension), and uncoupled or coupled with Frobenius norm or nuclear norm."],[450,90]) 

Total_Reg_parameter=Parameterchoice(Reconstructionframe,pos,'Global regularization: ','1')
tip = ListboxToolTip(Total_Reg_parameter.frame, ["Chooses an overall regularization parameter, which affects all channels uniformly (each weight is multiplied by this value), so if a suitable ratio of weights is found, it suffices to adapt this parameter."],[300,120]) 

pos +=1
Discrepancychoice=Optionbox(Reconstructionframe,['KL','L2'],pos,'Discrepancy:     ',0)
tip = ListboxToolTip(Discrepancychoice.frame, ["Chooses which discrepancy type to use: KL (Kullback-Leibler) is suited for data affected by Poisson noise, L2 (sum of squares) is suited for data affected by Gaussian noise."],[450,90]) 
 
Maxiterchoice=Parameterchoice(Reconstructionframe,pos,'Number of iterations: ','2500') 
tip = ListboxToolTip(Maxiterchoice.frame, ["Fixes the number of iterative steps used in the reconstruction algorithm."],[300,45]) 

pos +=1
Automaticbutton=Button(Reconstructionframe,text='Automatic weight choice',command=proposed_weightratios)
Automaticbutton.grid(row=pos,column=3,sticky='E')
tip = ListboxToolTip(Automaticbutton, ["This feature tries to find a suitable ratio for channel weights by considering the mean values of the respective data. Then, only the global regularization parameter needs to be adapted. Though the determined weights might not be perfect, they are nevertheless a good starting point for manual fine tuning."],[450,105]) 


#Computationaloption
pos=1
Computational_Frameall=Frame(fenster,borderwidth=4,width=10000,height=100000, pady=10)
Computational_Frameall.pack(fill=X)
Computational_Framebig=Frame(Computational_Frameall,borderwidth=4,relief="groove",width=10000,height=100000)
Computational_Framebig.pack(fill=X)
Computational_Frame= Frame(Computational_Frameall,borderwidth=4,relief="sunken",width=10000,height=100000)
Computational_Frame.pack(fill=X)
Computational_Frametitle=Label(Computational_Framebig,text="Computational options", font=headingFont)
Computational_Frametitle.pack()

Computational_frame_plotandoverlaying=Frame(Computational_Frame)
Computational_frame_plotandoverlaying.grid(row=pos,columnspan=4)

verticalpos=0
Plotbutton=Checkbuttons(Computational_frame_plotandoverlaying,pos,verticalpos,'Plot frequency','','0.05')
tip = ListboxToolTip(Plotbutton.frame, ["When activated, the program will plot preliminary results during the reconstruction process. In the field, a number can be entered that determines the frequency of plot updates. For numbers N>1, this refers to an update every N iterations, for N in [0,1] this refers to the ratio with respect to the total number of iterations (0.05=5%)."],[400,120]) 

verticalpos+=2
Overlappingbutton=Checkbuttons(Computational_frame_plotandoverlaying,pos,verticalpos,'Overlapping slices','','1')
tip = ListboxToolTip(Overlappingbutton.frame, ["When computations require more memory than available on the OpenCL device, the code tries to split the selected slices into subsections and reconstructs them individually. To still observe a coupling of the sections, one can overlap these subsection by a given amount of slices. If not checked, the value is 0 (no overlap)."],[400,120]) 

pos+=2

slices_choice=Slicereader(Computational_Frame, pos,'Slices to process:   ')
tip = ListboxToolTip(slices_choice.frame, ["Determines which slices will be reconstructed. The initial slice, the final one and a stepsize between the slices can be given. (E.g., 0 10 2 means that the slices 0,2,4,6,8,10 are reconstructed."],[450,90]) 

pos+=2

Extractionframe=Frame(fenster)
Extractionframe.pack(side=BOTTOM)
Extract_InformationButton=Button(Extractionframe,text='Show command line',command=Extract_Information_show)
tip = ListboxToolTip(Extractionframe, ["Shows the command line corresponding to the current options. The field on the left specifies the path and prefix of the output files. (This has no impact on the reconstructions started via the GUI.)"],[450,70],[-20,-100]) 
Extract_InformationButton.grid(row=0,column=2)

Extract_Information_outfield=Outline(Extractionframe)


#Try to load pyopencl and get information on devices
try:
	my_gpu_devices=['No device found']
	GPUchoice=Optionbox(Computational_Frame,my_gpu_devices,pos,'',0)
	import pyopencl as cl
	platforms = cl.get_platforms()
	my_gpu_devices = []
	try:
		for platform in platforms:
			my_gpu_devices.extend(platform.get_devices())
		gpu_devices = [device for device in my_gpu_devices if device.type == cl.device_type.GPU]
		non_gpu_devices = [device for device in my_gpu_devices if device not in gpu_devices]
		my_gpu_devices = gpu_devices + non_gpu_devices
	except:
		print('No OpenCL devices found.')
        		
	if my_gpu_devices==[]:
		my_gpu_devices=['No device found']
		raise cl.Error('No device found, make sure PyOpenCL was installed correctly.')
	GPUchoice.frame.grid_forget()
	del GPUchoice
	GPUchoice=Optionbox(Computational_Frame,my_gpu_devices,pos,'',0)
	tip = ListboxToolTip(GPUchoice.frame, ["Chooses which OpenCL device is used for the computations."],[450,30]) 

except ImportError as err:
	template = "An exception of type {0} occurred. Arguments:\n{1!r}"
	message = template.format(type(err).__name__, err.args)
	eprint(message)
	
	
	Returnwindow=Toplevel()
	Returnwindow.title("Warning!!! Unable to load PyOpenCL.")
	
	
	button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy)
	button.pack()	
	
	ErrorLabel=Label(Returnwindow,text='A Critical Error has occured as Python was unable to load PyOpenCL: Make sure the PyOpenCL is correctly installed and configured with respect to the available hardware.')
	ErrorLabel.pack()
	
	scrollbar = Scrollbar(Returnwindow)
	scrollbar.pack(side=RIGHT, fill=Y)
	listbox = Text(Returnwindow, yscrollcommand=scrollbar.set)
	listbox.insert(END,repr(err))
	listbox.pack(side=LEFT, fill=BOTH)
	listbox.config(state=DISABLED)
	scrollbar.config(command=listbox.yview)

except cl.Error as err:
	template = "An exception of type {0} occurred.  Arguments:\n{1!r}"
	message = template.format(type(err).__name__,err.args)
	eprint(message)
	my_gpu_devices=['No device found']
	
	Returnwindow=Toplevel()
	Returnwindow.title("Warning!!! PyOpenCL could not find a suitable device.")
	
	button = Button(Returnwindow, text="Dismiss", command=Returnwindow.destroy)
	button.pack()	
	
	
	
	ErrorLabel=Label(Returnwindow,text='A Critical Error has occured as Python was unable to find a suitable device:')
	ErrorLabel.pack()
	
	scrollbar = Scrollbar(Returnwindow)
	scrollbar.pack(side=RIGHT, fill=Y)
	listbox = Text(Returnwindow, yscrollcommand=scrollbar.set)
	listbox.insert(END,repr(err))
	listbox.pack(side=LEFT, fill=BOTH)
	listbox.config(state=DISABLED)
	scrollbar.config(command=listbox.yview)
	
# First computationbutton



computationbuttonframeall=Frame(fenster,borderwidth=4,width=10000,height=100000, pady=10)
computationbuttonframeall.pack(fill=X)

computationbuttonframetitle=Frame(computationbuttonframeall,borderwidth=4,relief="groove",width=10000,height=100000)
computationbuttonframetitle.pack(side=TOP,fill=X)
computationbuttontitle=Label(computationbuttonframetitle,text="Reconstructions", font=headingFont)
computationbuttontitle.pack()


computationbuttonframeBig=VerticalScrolledFrame(computationbuttonframeall,10000,borderwidth=4,relief="sunken",width=10000,height=100000)
computationbuttonframeBig.pack(side=BOTTOM,fill=X)


global initial_computationbutton;#Serves as anchor for all computations
initial_computationbutton=computationbutton(None,computationbuttonframeBig.interior)

def main():	
	global startingpath
	startingpath=os.getcwd()
	mypath= os.path.realpath(__file__)
	mypath= '/'.join( mypath.split('/')[:-1] )
	os.chdir(mypath)
	
	##Debugging Mode	
	#import pdb; pdb.set_trace()
	
	
	
	#Cleanup old tmp files
	Filelist=glob.glob('tmp/*')
	for entry in Filelist:#Delete files in tmp folder which are older than 5 days
		if time.time()-os.stat(entry).st_mtime>60*60*24*5:
		#	shutil.rmtree(entry)
			pass
	#Main loop
	w, h = fenster.winfo_screenwidth(), fenster.winfo_screenheight()
	fenster.geometry("%dx%d+0+0" % (w, h))
	fenster.mainloop()
	

if __name__ == '__main__':
	main()  


# flake8: noqa
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
import os
import osyris
from scipy.stats import binned_statistic
from tqdm import tqdm


class suppress_stdout(object):
	"""
	Bit of code to supress stdout output of osiris when reading data
	"""
	def __init__(self,stdout = None, stderr = None):
		self.devnull = open(os.devnull,'w')
		self._stdout = stdout or self.devnull or sys.stdout
		self._stderr = stderr or self.devnull or sys.stderr

	def __enter__(self):
		self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
		self.old_stdout.flush(); self.old_stderr.flush()
		sys.stdout, sys.stderr = self._stdout, self._stderr

	def __exit__(self, exc_type, exc_value, traceback):
		self._stdout.flush(); self._stderr.flush()
		sys.stdout = self.old_stdout
		sys.stderr = self.old_stderr
		self.devnull.close()


def relative_sort(list1, list2):
	"""
	Sorts lists 1 and 2 based on the values of list1
	"""
	tuples = zip(*sorted(zip(list1, list2)))
	l1, l2 = [list(tuple) for tuple in tuples]
	l1 = np.asarray(l1); l2 = np.asarray(l2)
	return l1, l2


def bin_data(rmin, rmax, nr, mydata_field1, mydata_field2):
	"""
	Overlays a log10 mean profile over field2 along field1
	rmin and rmax in log10 units
	"""

	# Radial bin edges and centers
	re = np.linspace(rmin,rmax,nr+1)
	log_field1 = np.zeros([nr])
	for i in range(nr):
		log_field1[i] = 0.5*(re[i]+re[i+1])
	# Modify r values so that the central cell is not "-inf"
	field1 = np.where(np.isinf(mydata_field1.values),-2.0,mydata_field1.values)
	# Bin the data in radial bins
	z0, edges = np.histogram(field1, bins=re)
	z1, edges = np.histogram(field1, bins=re, weights=mydata_field2.values)
	mean_profile = np.log10(z1 / z0)
	return log_field1, mean_profile


def rotation_matrix(vec, angle):
    R = np.cos(angle)*np.identity(3) + (np.sin(angle))*np.cross(vec, np.identity(vec.shape[0]) * -1) +(1-np.cos(angle))*(np.outer(vec,vec))
    return R


def gamma_eos(dens):
	"""
	Return effective adiabatic index according to density (in cgs) (fig 1. Bhandare et al. 2020)
	"""
	if dens < 1e-13:
		return 1
	if dens < 1e-11:
		return 5/3
	if dens < 1e-8:
		return 7/5
	if dens < 1e-5:
		return 1.1
	if dens > 1e-5:
		return 5/3


def compute_radial_pos_vel(thread):
	"""
	Simply compute radial velocity & radius and store results in "hydro" group of thread
	"""
	ind = np.argmax(thread.ramses_data["hydro"]["density"])
	center = thread.ramses_data["amr"]["position"][ind.values]

	centered_pos = thread.ramses_data["amr"]["position"] - center
	radial_vel = (thread.ramses_data["hydro"]["velocity"].x * centered_pos.x + thread.ramses_data["hydro"]["velocity"].y * centered_pos.y + thread.ramses_data["hydro"]["velocity"].z * centered_pos.z)/centered_pos.norm # (vx*x+vy*y+vz*z)/r
	radius = centered_pos.norm

	return radius, radial_vel


def compute_COM(thread, condition=None):
	"""
	Compute and return center of mass.
	"""
	if condition is not None:
		M = thread.ramses_data["hydro"]["mass"][condition]
		P = thread.ramses_data["amr"]["position"][condition]
	else:
		M = thread.ramses_data["hydro"]["mass"]
		P = thread.ramses_data["amr"]["position"]
	mixi = np.sum(M*P.x)
	miyi = np.sum(M*P.y)
	mizi = np.sum(M*P.z)
	mi = np.sum(M)
	COM_x = mixi/mi
	COM_y = miyi/mi
	COM_z = mizi/mi
	COM = osyris.Vector(x=COM_x.to("cm").values, y=COM_y.to("cm").values, z=COM_z.to("cm").values, unit="cm")

	return COM


def extract_spherical_shell(thread, radius, center):
	"""
	Extract a spherical shell from dataset.
	radius must be an osyris Array with a single element
	"""

	X = (thread.ramses_data["amr"]["position"].x - center.x).to("au").values
	Y = (thread.ramses_data["amr"]["position"].y - center.y).to("au").values
	Z = (thread.ramses_data["amr"]["position"].z - center.z).to("au").values

	dx = thread.ramses_data["amr"]["dx"].to("au").values

	vertices = np.asarray([[X - .5*dx, Y - .5*dx, Z - .5*dx],
						[X - .5*dx, Y - .5*dx, Z + .5*dx],
						[X - .5*dx, Y + .5*dx, Z - .5*dx],
						[X - .5*dx, Y + .5*dx, Z + .5*dx],
						[X + .5*dx, Y + .5*dx, Z - .5*dx],
						[X + .5*dx, Y + .5*dx, Z + .5*dx],
						[X + .5*dx, Y - .5*dx, Z - .5*dx],
						[X + .5*dx, Y - .5*dx, Z + .5*dx]])

	vertices_radius = np.asarray([np.linalg.norm(v,axis=0) for v in vertices])

	vertices_inside_sphere = np.asarray([v < radius.to("au").values for v in vertices_radius])
	# when selecting cells, we need at least one inside the sphere, and one outside:
	cell_selection = np.logical_and(np.any(vertices_inside_sphere, axis=0), ~np.all(vertices_inside_sphere, axis=0))
	return cell_selection

def core_info(log_dir):

	logs = os.listdir(log_dir) #Probing log_dir
	logs = [log for log in logs if 'max_rho_T' in log]

	rhoc = []; Tc = []; time = []; dt = []
	for file in logs: #reading log_dir data
		filename = log_dir+file
		data = open(filename,'r').readlines()
		for line in data:
			l = line.split()
			rhoc.append(float(l[1]))
			Tc.append(float(l[2]))
			time.append(float(l[0]))
	rhoc = np.asarray(rhoc)
	Tc = np.asarray(Tc)
	time = np.asarray(time)
	t = np.copy(time)
	time,rhoc = relative_sort(t, rhoc)
	_,Tc = relative_sort(t, Tc)


	print("Gathered {} data points from {} output files.".format(Tc.size, len(logs)))
	print("Max density: {} [g/cm^3]".format(rhoc.max()))
	print("Max temperature: {} [K]".format(Tc.max()))
	print("Time lapse: {:.3f} [kyr] ({:.3f} - {:.3f} [kyr])".format(time[-1]-time[0],time[-1],time[0]))

	return time, rhoc, Tc


from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

def define_scalebar(ax, im_resolution, real_bar_size, real_im_size):
	npix_bar = int(im_resolution*real_bar_size.to("R_sol").magnitude/(real_im_size.to("R_sol").magnitude)) # select as many pixels as needed for the bar to be real_bar_size
	# real_bar_size = npix_bar * real_im_size/im_resolution
	fontprops = fm.FontProperties(size=16)
	rbs = real_bar_size.to("au").magnitude
	return AnchoredSizeBar(ax.transData, npix_bar, '{0}'.format(rbs)+' '+r'$\mathrm{AU}$',
						   'lower center', pad=0.1, color='white',frameon=False, size_vertical=3, fontproperties=fontprops)


def average_in_bin(thread, scalar, unit, condition=None, rr=None):
	"""
	Average a scalar quantity in radial bins. May take a condtional array
	"""
	if "radial vel" not in thread.ramses_data["hydro"]:
		r,vr = compute_radial_pos_vel(thread)
		thread.ramses_data["hydro"]["radial vel"] = vr.to("km/s")
		thread.ramses_data["hydro"]["radius"] = r
	if rr is None: # define rr if user did not provide it
		rmin = 5e-4
		rmax = np.max(thread.ramses_data["hydro"]["radius"].max().to("au").values)
		rr = np.logspace(np.log10(rmin), np.log10(rmax), 200)
	scalars = []
	if condition is None:
		r = thread.ramses_data["hydro"]["radius"].to("au").values
		scalar_array = thread.ramses_data["hydro"][scalar].to(unit).values
	else:
		r = thread.ramses_data["hydro"]["radius"].to("au").values[condition]
		scalar_array = thread.ramses_data["hydro"][scalar].to(unit).values[condition]
	for i in tqdm(range(rr.size), ascii=True, desc="Binning " + scalar.replace("_{}", "")):
		if i != 0:
			c = (r <= rr[i]) & (r > rr[i-1])
		else:
			c = (r <= rr[i])
		if (np.sum(c) == 0):
			scalars.append(np.nan)
		else:
			scalars.append(np.average(scalar_array[c], weights=None))
	return rr, np.asarray(scalars)

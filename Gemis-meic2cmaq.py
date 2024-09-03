# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:10:09 1508

@author: Dylan Dingyuan Liang
@email: dliangad@connect.ust.hk
"""

import pandas as pd
import os
import netCDF4 as nc
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from glob import glob
import xesmf as xe
from calendar import monthrange

def get_datetime(ds):
    df = pd.DataFrame(ds[:][:,0,:])
    datelist = pd.to_datetime(df.iloc[:,0].astype(str)+df.iloc[:,1].astype(str).str.zfill(6),format='%Y%j%H%M%S')
    return(datelist)

def read_meic(files):
    species_names = [os.path.splitext(os.path.basename(file))[0].split('_')[-1] for file in files]
    data_arrays = []
    for i, (file, species_name) in enumerate(zip(files, species_names)):
        ds_specie = xr.open_dataset(file)
        ds_specie = ds_specie.assign_coords(species=species_name)
        data_arrays.append(ds_specie)
    
    ds = xr.concat(data_arrays, dim='species') 
    
    valid = ds.z != ds.z.attrs['nodata_value']
    ds['z'] = ds['z'].where(valid, 0.)
    
    dims = ds['dimension'].values
    ydim = dims[0, 1]
    xdim = dims[0, 0]
    
    emi_lon_b = np.arange(ds['x_range'][0, 0],
                          ds['x_range'][0, 1] + ds['spacing'][0, 0],
                          ds['spacing'][0, 0])
    emi_lat_b = np.arange(ds['y_range'][0, 0],
                          ds['y_range'][0, 1] + ds['spacing'][0, 1],
                          ds['spacing'][0, 1])
    emi_lon = (emi_lon_b[:-1] + emi_lon_b[1:]) / 2
    emi_lat = (emi_lat_b[:-1] + emi_lat_b[1:]) / 2
    lon2d, lat2d = np.meshgrid(emi_lon, emi_lat)
            
    # assign to ds
    ds['longitude'] = xr.DataArray(lon2d,
                                   coords=[emi_lat, emi_lon],
                                   dims=['y', 'x'])
    ds['latitude'] = xr.DataArray(lat2d,
                                  coords=[emi_lat, emi_lon],
                                  dims=['y', 'x'])
    
    dds = xr.DataArray(ds['z'].values, dims=['species', 'xy']).astype(np.float64)
    flag = xr.DataArray(dds.values.reshape((-1, ydim, xdim), order='C'),
                        dims=['species', 'y', 'x'],
                        attrs=ds['z'].attrs
                        )
    # drop old dims
    ds = ds.drop_dims(['xysize', 'side'])
    ds['z'] = flag
    return ds

def gen_vnames(unames, subfix="", numc=16):
    namefix = [f"{uname}{subfix}" for uname in unames]
    namefix = [f"{namefix[i]:<16}" for i in range(len(namefix))]
    return "".join(namefix)

def GenCMAQ(GDCRO, MTCRO, BSEMIS, EMISFILE, SCALEFAC):
    TFG = MTCRO.variables["TFLAG"]
    cmaqtime = get_datetime(TFG)
    STDATE = cmaqtime[0]
    THOUR = len(MTCRO.dimensions["TSTEP"])
    ROWS = len(GDCRO.dimensions["ROW"])
    COLS = len(GDCRO.dimensions["COL"])
    LAYS = MTCRO.NLAYS
    TSTP = THOUR
    NEMI = len(BSEMIS) # sectors
    NV = len(VARBASEEMI) # meic pollutants
    NVCMAQ = len(VNAMS) # cmaq species
    MCIPLAYS = MTCRO.VGLVLS.tolist()

    print("Grid Info: Start Date:", STDATE, "THour:", THOUR, "TLay:", LAYS, "COL:", COLS, "ROW:", ROWS, ".")
    
    # ------------- ------------- ------------- #
    days = monthrange(yyyy_emi, mm)[1]
    emi_path = data_path_emi+str(yyyy_emi)+'_'+str(mm).zfill(2)
    
    REMI = np.zeros((COLS, ROWS, NEMI, NV))
    sectors = ["agriculture", "industry", "power", "residential", "transportation"] # should be order consistent with BSEMIS
    for index_s,s in enumerate(sectors):
        pattern = emi_path + '_' + s + '*' + '.nc'
        files = sorted(glob(pattern))
        em = read_meic(files)
        
        # -------------
        lat = GDCRO.variables['LAT'][0, 0, :, :]
        lon = GDCRO.variables['LON'][0, 0, :, :]
        
        grid_in = {'lon': em['longitude'].values, 'lat': np.flip(em['latitude'].values, axis=0)}
        grid_out = {'lon':  lon.data, 'lat':lat.data}
        
        # spatial interpolation 
        regridder = xe.Regridder(grid_in, grid_out, 'nearest_s2d', reuse_weights=False)
        emi = regridder(em['z'])
        
        for VN, varname in enumerate(VARBASEEMI):
            REMI[:, :, index_s, VN] = np.transpose(emi.sel(species=varname).values) * SCALEFAC[index_s]
    
    REMM = REMI.copy() # A dunplication of raw emis data.
    REMH = np.zeros((COLS, ROWS, TSTP, NEMI, NV), dtype=np.float32)
    for EMI in range(NEMI):
        # REMM[:, :, EMI, :] = REMM[:, :, EMI, :] * TPROM[BSEMIS[EMI]][STMO] / DayMon[STMO] # Yearly to Daily Emission
        REMM[:, :, EMI, :] = REMM[:, :, EMI, :] / days # Monthly to Daily Emission

        ttag=1
        for TI in range(1, TSTP):
            # Loop through emissions
            for EMI in range(NEMI):
                REMH[:, :, TI - 1, EMI, :] = REMM[:, :, EMI, :] * TPROH[BSEMIS[EMI]][ttag - 1] # Hourly Emission
            ttag += 1
    REMH = REMH.transpose(2, 1, 0, 3, 4) # (182, 138, 25, 5, 7) to (25, 138, 182, 5, 7)
    ncvarlist = gen_vnames(VNAMS + ["PMC"])
    
    # ------
    # Vertical and speciation
    # Create a NetCDF file
    EMIS3D = Dataset(EMISFILE, 'w', format='NETCDF4')
    # Define dimensions
    EMIS3D.createDimension('TSTEP', TSTP)  # 0
    EMIS3D.createDimension('DATE-TIME', 2)  # 1
    EMIS3D.createDimension('LAY', LAYS)    # 2
    EMIS3D.createDimension('VAR', NVCMAQ + 1)  # 3
    EMIS3D.createDimension('ROW', ROWS)  # 4
    EMIS3D.createDimension('COL', COLS)  # 5
    # Copy global attributes from GDCRO
    EMIS3D.setncatts(GDCRO.__dict__)
    # Add or update specific global attributes
    EMIS3D.setncattr('EXEC_ID', 'MEIC2CMAQ')
    EMIS3D.setncattr('TSTEP', 10000)
    EMIS3D.setncattr('NLAYS', LAYS)
    EMIS3D.setncattr('NVARS', NVCMAQ + 1)
    EMIS3D.setncattr('VGLVLS', MCIPLAYS[0:(LAYS + 1)])
    EMIS3D.setncattr('VAR-LIST', ncvarlist)
    EMIS3D.setncattr('FILEDESC', 'CMAQ Emission File by Liang Dingyuan') # MEIC2CMAQ
    
    # ------
    # Generating TFLAG, not finished
    TFG = np.zeros((2, NVCMAQ + 1, TSTP), dtype=np.int32)
    ttag = 12 # 0
    dtag = 1
    for TI in range(1, TSTP + 1):
        if ttag > 23:
            ttag = 0
            dtag += 1
        cday = STDATE + pd.DateOffset(days=dtag)
        cjul = cday.strftime('%Y%j')
        ctim = ttag * 10000
        ttag += 1
        TFG[0, :, TI - 1] = int(cjul)
        TFG[1, :, TI - 1] = int(ctim)
    TFG = TFG.transpose(2, 1, 0)
    # Define TFLAG variable
    TFLAG = EMIS3D.createVariable('TFLAG', np.int32, ('TSTEP', 'VAR', 'DATE-TIME'))
    TFLAG.long_name = "TFLAG"
    TFLAG.units = "<YYYYDDD,HHMMSS>"
    TFLAG.var_desc = "Timestep-valid flags:  (1) YYYYDDD or (2) HHMMSS"
    # Set the TFLAG values
    TFLAG[:, :, :] = TFG
    
    # ------
    # Process VOCs
    for SPC in range(len(VOCN['VNAM'])):
        ncvar = VOCN['VNAM'][SPC]
        VN = VARBASEEMI.index('VOC')
        tmpval = np.zeros((TSTP, LAYS, ROWS, COLS), dtype=np.float32)
        for EMI in range(len(BSEMIS)):
            if BSEMIS[EMI][:3] == "IND":
                for LY in range(len(LF_IND)):
                    tmpval[:, LY, ...] += REMH[..., EMI, VN] * LF_IND[LY] * VPRO[BSEMIS[EMI]][SPC]  # 4D Emission
            elif BSEMIS[EMI][:3] == "POW":
                for LY in range(len(LF_POW)):
                    tmpval[:, LY, ...] += REMH[..., EMI, VN] * LF_POW[LY] * VPRO[BSEMIS[EMI]][SPC]  # 4D Emission
            else:
                tmpval[:, 0, ...] += REMH[..., EMI, VN] * VPRO[BSEMIS[EMI]][SPC]  # 4D Emission
        
        try:
            VN = VARBASEEMI.index(ncvar)
            for EMI in range(len(BSEMIS)):
                if BSEMIS[EMI][:3] == "IND":
                    for LY in range(len(LF_IND)):
                        tmpval[:, LY, ...] += REMH[:, :, :, EMI, VN] * LF_IND[LY]
                elif BSEMIS[EMI][:3] == "POW":
                    for LY in range(len(LF_POW)):
                        tmpval[:, LY, ...] += REMH[:, :, :, EMI, VN] * LF_POW[LY]
                else:
                    tmpval[:, 0, ...] += REMH[:, :, :, EMI, VN]
        except ValueError:
            pass
    
        # merged iVOC
        print("Generating", ncvar, "...")
        molmas = VOCN['MMAS'][SPC] # molecule weight
        # Create the variable in the NetCDF file
        VOC_var = EMIS3D.createVariable(ncvar, np.float32, ('TSTEP', 'LAY', 'ROW', 'COL'))
        # VOC_var = EMIS3D.variables[ncvar]
        VOC_var.long_name = ncvar
        VOC_var.units = "moles/s"
        VOC_var.var_desc = "Model species " + ncvar
        # Convert and set values in the variable
        tmpval = tmpval * 1000 * 1000 / molmas / 3600  # mol/s
        VOC_var[:, :, :, :] = tmpval
    
    # ------
    # Process NOx
    VN = VARBASEEMI.index('NOx')
    for SPC in range(len(NOXN['VNAM'])):
        tmpval = np.zeros((TSTP, LAYS, ROWS, COLS), dtype=np.float32)
        for EMI in range(len(BSEMIS)):
            if BSEMIS[EMI][:3] == "IND":
                for LY in range(len(LF_IND)):
                    tmpval[:, LY, ...] += REMH[..., EMI, VN] * LF_IND[LY] * NPRO[BSEMIS[EMI]][SPC]
            elif BSEMIS[EMI][:3] == "POW":
                for LY in range(len(LF_POW)):
                    tmpval[:, LY, ...] += REMH[..., EMI, VN] * LF_POW[LY] * NPRO[BSEMIS[EMI]][SPC]
            else:
                tmpval[:, 0, ...] += REMH[..., EMI, VN] * NPRO[BSEMIS[EMI]][SPC]
        #merged iVOC
        ncvar = NOXN['VNAM'][SPC]
        print("Generating", ncvar, "...")
        molmas = NOXN['MMAS'][SPC]
    
        # Create the variable in the NetCDF file
        NOx_var = EMIS3D.createVariable(ncvar, np.float32, ('TSTEP', 'LAY', 'ROW', 'COL'))
        NOx_var.long_name = ncvar
        NOx_var.units = "moles/s"
        NOx_var.var_desc = "Model species " + ncvar
    
        # Convert and set values in the variable
        tmpval = tmpval * 1000 * 1000 / molmas / 3600  # mol/s
        NOx_var[:, :, :, :] = tmpval
    
    # ------
    # Process PMFine
    for SPC in range(len(PMFN['VNAM'])):
        ncvar = PMFN['VNAM'][SPC]
        VN = VARBASEEMI.index('PM25')
        tmpval = np.zeros((TSTP, LAYS, ROWS, COLS), dtype=np.float32)
        for EMI in range(len(BSEMIS)):
            if BSEMIS[EMI][:3] == "IND":
                for LY in range(len(LF_IND)):
                    tmpval[:, LY, ...] += REMH[:, :, :, EMI, VN] * LF_IND[LY] * PPRO[BSEMIS[EMI]][SPC]
            elif BSEMIS[EMI][:3] == "POW":
                for LY in range(len(LF_POW)):
                    tmpval[:, LY, ...] += REMH[:, :, :, EMI, VN] * LF_POW[LY] * PPRO[BSEMIS[EMI]][SPC]
            else:
                tmpval[:, 0, ...] += REMH[:, :, :, EMI, VN] * PPRO[BSEMIS[EMI]][SPC]
        
        if ncvar == 'PEC':
            varmeic = 'BC'
        elif ncvar == 'POC':
            varmeic = 'OC'
        else:
            varmeic = None
        
        try:
            VN = VARBASEEMI.index(varmeic)
            for EMI in range(len(BSEMIS)):
                if BSEMIS[EMI][:3] == "IND":
                    for LY in range(len(LF_IND)):
                        tmpval[:, LY, ...] += REMH[:, :, :, EMI, VN] * LF_IND[LY]
                elif BSEMIS[EMI][:3] == "POW":
                    for LY in range(len(LF_POW)):
                        tmpval[:, LY, ...] += REMH[:, :, :, EMI, VN] * LF_POW[LY]
                else:
                    tmpval[:, 0, ...] += REMH[:, :, :, EMI, VN]
        except ValueError:
            pass
        
        print("Generating", ncvar, "...")
        var = EMIS3D.createVariable(ncvar, np.float32, ('TSTEP', 'LAY', 'ROW', 'COL'))
        var.long_name = ncvar
        var.units = "g/s"
        var.var_desc = "Model species " + ncvar
    
        tmpval = tmpval * 1000 * 1000 / 3600  # g/s
        var[:, :, :, :] = tmpval
    
    # ------
    # Process OTHER GAS
    for SPC in range(len(OTHN['VNAM'])):
        ncvar = OTHN['VNAM'][SPC]
        VN = VARBASEEMI.index(ncvar)
        tmpval = np.zeros((TSTP, LAYS, ROWS, COLS), dtype=np.float32)
        for EMI in range(len(BSEMIS)):
            if BSEMIS[EMI][:3] == "IND":
                for LY in range(len(LF_IND)):
                    tmpval[:, LY, ...] += REMH[..., EMI, VN] * LF_IND[LY]
            elif BSEMIS[EMI][:3] == "POW":
                for LY in range(len(LF_POW)):
                    tmpval[:, LY, ...] += REMH[..., EMI, VN] * LF_POW[LY]
            else:
                tmpval[:, 0, ...] += REMH[..., EMI, VN]
    
        print("Generating", ncvar, "...")
        molmas = OTHN['MMAS'][SPC]
        # Create the variable in the NetCDF file
        other_gas_var = EMIS3D.createVariable(ncvar, np.float32, ('TSTEP', 'LAY', 'ROW', 'COL'))
        other_gas_var.long_name = ncvar
        other_gas_var.units = "moles/s"
        other_gas_var.var_desc = "Model species " + ncvar
    
        # Convert and set values in the variable
        tmpval = tmpval * 1000 * 1000 / molmas / 3600  # mol/s
        other_gas_var[:, :, :, :] = tmpval
    
    # ------
    # Process PMC
    VN_PMC = VARBASEEMI.index('PM10')
    VN_PMF = VARBASEEMI.index('PM25')
    tmpval = np.zeros((TSTP, LAYS, ROWS, COLS), dtype=np.float32)
    for EMI in range(len(BSEMIS)):
        if BSEMIS[EMI][:3] == "IND":
            for LY in range(len(LF_IND)):
                tmpval[:, LY, ...] += (REMH[:, :, :, EMI, VN_PMC] * LF_IND[LY] - REMH[:, :, :, EMI, VN_PMF] * LF_IND[LY])
        elif BSEMIS[EMI][:3] == "POW":
            for LY in range(len(LF_POW)):
                tmpval[:, LY, ...] += (REMH[:, :, :, EMI, VN_PMC] * LF_POW[LY] - REMH[:, :, :, EMI, VN_PMF] * LF_POW[LY])
        else:
            tmpval[:, 0, ...] += (REMH[:, :, :, EMI, VN_PMC] - REMH[:, :, :, EMI, VN_PMF])
    
    ncvar = "PMC"
    print("Generating", ncvar, "...")
    var = EMIS3D.createVariable(ncvar, np.float32, ('TSTEP', 'LAY', 'ROW', 'COL'))
    var.long_name = ncvar
    var.units = "g/s"
    var.var_desc = "Model species " + ncvar
    tmpval = tmpval * 1000 * 1000 / 3600  # g/s
    var[:, :, :, :] = tmpval
    
    # Close the NetCDF files
    EMIS3D.close()
    print("Done.")

# ------------- ------------- ------------- ------------- #
print("-----------------------------")
print("This program is developed to generate CMAQ-ready emission files based on the MEIC inventory (http://meicmodel.org.cn/#firstPage).")
print("Modifications can be made to use the original VOC speciate information of MEIC.")
print("Please note that this program is open-source and is provided 'as-is', without any warranty or guarantee of accuracy.")
print("Users are encouraged to review and modify the code as needed to suit their specific requirements.")
print("For assistance or inquiries, feel free to contact me at dliangad@connect.ust.hk.")
print("-----------------------------")

data_path_emi = 'D:/EMISSIONs/MEIC_CB05_202007/'
yyyy_emi = 2020
mm = 7

gddir = "D:/EMISSIONs/"
GDCRO = nc.Dataset(gddir + "GRIDCRO2D_27km.20220704.nc")
MTCRO = nc.Dataset(gddir + "METCRO3D_27km.20220704.nc")

VARBASEEMI = ['BC', 'ALD2', 'ALDX', 'CH4', 'ETH', 'ETHA', 'ETOH', 'FORM', 'IOLE', 'ISOP', 'MEOH', 'NVOL',
              'OLE', 'PAR', 'TERP', 'TOL', 'UNR', 'XYL', 'CO', 'NH3', 'NOx', 'OC', 'PM10', 'PM25', 'SO2', 'VOC'] # meic pollutants

# VARBASEEMI = ["VOC", "PM25", "NOx", "CO", "SO2", "NH3", "CH4", "ALD2", "ALDX", "ETH", "ETOH", "FORM", "IOLE", "ISOP", "MEOH", "NVOL", "TERP", "PM10"] # meic pollutants

# ------ VOC ------ #
VOCN = pd.DataFrame({
    "VNAM": ["XYL", "UNR", "TOL", "PAR", "OLE", "ETHA", "BENZENE"],
    "MMAS": [108.9851, 30.2036, 97.6694, 16.8563, 28.0532, 30.0690, 78.1118]
})

LF_IND = [0.7, 0.2, 0.1, 0] # vertical allocation
LF_POW = [0.2, 0.4, 0.3, 0.1]

VPRO = pd.DataFrame({
    "AGR": [0.1436, 0.1347, 0.1335, 0.3746, 0.0356, 0.1781, 0.0378],
    "IND": [0.1436, 0.1347, 0.1335, 0.3746, 0.0356, 0.1781, 0.0378],
    "POW": [0.1436, 0.1347, 0.1335, 0.3746, 0.0356, 0.1781, 0.0378],
    "RES": [0.1436, 0.1347, 0.1335, 0.3746, 0.0356, 0.1781, 0.0378],
    "TRA": [0.1436, 0.1347, 0.1335, 0.3746, 0.0356, 0.1781, 0.0378]
})

RVOC = ["ETHA", "OLE", "PAR", "TOL", "UNR", "XYL"] # reallocation pollutants

# ------ Fine particulate matter ------ #
PMFN = pd.DataFrame({
    "VNAM": ["PEC", "POC", "PSO4", "PNO3", "PCA", "PMG", "PK", "PFE", "PAL", "PSI", "PTI", "PMN", "PCL", "PNH4", "PNA", "PMOTHR", "PNCOM"]
})

PPRO = pd.DataFrame({
    "AGR": [0.01915, 0.02684, 0.12931, 0.00165, 0.03543, 0.00467, 0.00447, 0.02767, 0.05542, 0.09073, 0.00410, 0.00023, 0.00074, 0.00367, 0.00132, 0.59458, 0.04379],
    "IND": [0.01915, 0.02684, 0.12931, 0.00165, 0.03543, 0.00467, 0.00447, 0.02767, 0.05542, 0.09073, 0.00410, 0.00023, 0.00074, 0.00367, 0.00132, 0.59458, 0.04379],
    "POW": [0.01915, 0.02684, 0.12931, 0.00165, 0.03543, 0.00467, 0.00447, 0.02767, 0.05542, 0.09073, 0.00410, 0.00023, 0.00074, 0.00367, 0.00132, 0.59458, 0.04379],
    "RES": [0.01915, 0.02684, 0.12931, 0.00165, 0.03543, 0.00467, 0.00447, 0.02767, 0.05542, 0.09073, 0.00410, 0.00023, 0.00074, 0.00367, 0.00132, 0.59458, 0.04379],
    "TRA": [0.01915, 0.02684, 0.12931, 0.00165, 0.03543, 0.00467, 0.00447, 0.02767, 0.05542, 0.09073, 0.00410, 0.00023, 0.00074, 0.00367, 0.00132, 0.59458, 0.04379]
})

RPMF = ["BC", "OC"]  # reallocation pollutants

# ------ NOX ------ #
NOXN = pd.DataFrame({
    "VNAM": ["NO", "NO2", "HONO"],
    "MMAS": [30, 46, 47]
})

NPRO = pd.DataFrame({
    "AGR": [0.90, 0.082, 0.008],
    "IND": [0.90, 0.082, 0.008],
    "POW": [0.90, 0.082, 0.008],
    "RES": [0.90, 0.082, 0.008],
    "TRA": [0.85, 0.082, 0.058]
})

# ------ other gas ------ #
OTHN = pd.DataFrame({
    "VNAM": ["CO", "SO2", "NH3", "CH4", "ALD2", "ALDX", "ETH", "ETOH", "FORM", "IOLE", "ISOP", "MEOH", "NVOL", "TERP"],
    "MMAS": [28, 64, 17, 16, 44, 38, 28, 46, 30, 56, 68, 32, 16, 165]
})

# ------ ------ ------ #

# Variable names
VNAMS = list(VOCN["VNAM"]) + list(PMFN["VNAM"]) + list(NOXN["VNAM"]) + list(OTHN["VNAM"])

# Monthly
TPROM = pd.DataFrame({
    "AGR": np.array([0.0579, 0.0568, 0.062, 0.0839, 0.0988, 0.114, 0.108, 0.1185, 0.0854, 0.0679, 0.0776, 0.0691]),
    "IND": np.array([0.0768, 0.072, 0.0851, 0.0782, 0.079, 0.0864, 0.0773, 0.0787, 0.0836, 0.0852, 0.0976, 0.1]),
    "POW": np.array([0.0943, 0.0717, 0.0846, 0.0819, 0.0821, 0.0797, 0.087, 0.0896, 0.0756, 0.0759, 0.0849, 0.0926]),
    "RES": np.array([0.1826, 0.1316, 0.1054, 0.0533, 0.0491, 0.0475, 0.0491, 0.0491, 0.0475, 0.055, 0.0906, 0.1394]),
    "TRA": np.array([0.0834, 0.0834, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0834])
})

# Weekly
TPROW = pd.DataFrame({
    "AGR": np.array([0.14286, 0.14286, 0.14286, 0.14286, 0.14286, 0.14286, 0.14286]),
    "IND": np.array([0.16, 0.16, 0.16, 0.16, 0.16, 0.12, 0.08]),
    "POW": np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.13, 0.12]),
    "RES": np.array([0.14286, 0.14286, 0.14286, 0.14286, 0.14286, 0.14286, 0.14286]),
    "TRA": np.array([0.155, 0.155, 0.155, 0.155, 0.155, 0.125, 0.1])})

# Hourly
TPROH = pd.DataFrame({
    "AGR": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0],
    "IND": [0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.04, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06, 0.04, 0.03, 0.03, 0.03, 0.03],
    "POW": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04], 
    "RES": [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.07, 0.07, 0.04, 0.04, 0.03, 0.05, 0.05, 0.04, 0.03, 0.03, 0.04, 0.07, 0.07, 0.07, 0.07, 0.05, 0.02, 0.02], 
    "TRA": [0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.03, 0.02]
})

SCALEFAC = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
BASEEMIS = ["AGR", "IND", "POW", "RES", "TRA"]

# Directory creation
output_dir = "D:/EMISSIONs/emis_ready"; os.makedirs(output_dir, exist_ok=True)
emission_filename = 'D1_cmaq_emis.ncf' 

# GenCMAQ function call (make sure you have the function defined)
GenCMAQ(GDCRO, MTCRO, BSEMIS=BASEEMIS, EMISFILE=f"{output_dir}/{emission_filename}", SCALEFAC=SCALEFAC)

print("Program completed.")
print("-----------------------------")


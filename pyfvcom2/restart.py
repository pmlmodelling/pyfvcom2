"""Functions for handling FVCOM restart files"""

import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import subprocess
from Typing import Optional


def write_restart(template_file_path: str, output_path: str, data: dict, new_datetime: Optional[datetime] = None) -> None:
    """Write data to a FVCOM restart file in NetCDF4 format

    Args:
        template_file_path (str): Path to the FVCOM restart template file.
        output_path (str): Path to save the FVCOM restart file.
        data (dict): Data to write to the restart file.
        new_datetime (str, optional): New datetime string to use, format 'YYYY-MM-DD_HH:MM:SS'. Defaults to None.
    """
    # Build a list of vars to exclude
    var_list = list(data.keys())

    # If overwriting the time variables, add time, Itime and Times to the exclude list
    if new_datetime is not None:
        var_list.extend(['time', 'Itime', 'Times'])

    # If overwriting the time variables
    exclude_vars = ' '.join([f'{var},' for var in var_list])[:-1]  # Remove trailing comma
    
    # Use ncks to copy over everything from the template file except the variables listed in the data dict
    ncks_command = f'ncks -O -x -v {exclude_vars} {template_file_path} {output_path}'
    subprocess.run(ncks_command, shell=True, check=True)

    # Now append the new data to the output file. We will copy the attributes from the template file
    # and then write the new data.
    with Dataset(template_file_path, 'r') as template_ds, Dataset(output_path, 'a') as output_ds:

        # Update time variables time, Itime and times
        if new_datetime is not None:
            # Read in the units of the time variable in the template file
            template_time_var = template_ds.variables['time']
            template_time_units = template_time_var.units

            # Form a datetime object from template_time_units
            ref_time_str = template_time_units.split('since')[1].strip()
            ref_datetime = datetime.strptime(ref_time_str, '%Y-%m-%d %H:%M:%S')

            # Set new time values
            new_time = (new_datetime - ref_datetime).total_seconds() / 86400.0 # seconds in a day
            new_Itime = int(new_time)
            new_times = new_datetime.strftime(f'{new_datetime.strftime("%Y-%m-%d")}T00:00:00.000000')

            # Update the time variables in the output dataset
            output_ds.variables['time'][:] = np.asarray(new_time, dtype=template_ds.variables['time'].datatype)
            output_ds.variables['Itime'][:] = np.asarray(new_Itime, dtype=template_ds.variables['Itime'].datatype)
            output_ds.variables['Times'][:] = np.asarray(new_times, dtype=template_ds.variables['Times'].datatype)

        # Now write the data variables
        for var_name, var_data in data.items():
            # Create the variable in the output dataset
            template_var = template_ds.variables[var_name]
            output_var = output_ds.createVariable(
                var_name,
                template_var.datatype,
                template_var.dimensions,
                zlib=True,
                complevel=4
            )

            # Copy variable attributes
            for attr_name in template_var.ncattrs():
                output_var.setncattr(attr_name, template_var.getncattr(attr_name))

            # Convert the data type of var_data if necessary
            if var_data.dtype != np.dtype(template_var.datatype):
                var_data = var_data.astype(template_var.datatype)
            
            # Write the data
            output_var[:] = var_data

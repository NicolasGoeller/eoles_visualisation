import pandas as pd
import numpy as np
import os as os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from itertools import zip_longest

#### To-DO
# - Create Price-Production plot - is kind of implemented with stacked barplot and line
# - Multi: Compound section for transformation into GWh into load function
# - Implement color into vardesc_df and function settings

### Helpers for variable manipulation and visualisation

def _category_names(df_names):
    """Simplifies variable names from file input
    
    Cleans up string of column names to be and splits to list.

    Args:
        df_names (string): String of level-0 columns names to be.
    """
    
    ## Clean up column name string, lower case and split into list
    df_names = df_names.replace(" ->", "").replace("\n", "").lower().split(",")
    
    ## Define level-0 categories that are expected
    cats = ["generation","consumption","storage","charge","cost"]
    
    ## Replace column name with basic label if it contains it
    for id in range(len(df_names)):
        if any(s in df_names[id] for s in cats):
            df_names[id] = [s for s in cats if s in df_names[id]][0]
    
    return(df_names)

def _index_flatselect(cols, df):
    """Flatten list of column selection for multindex
    
    Takes a list of column selection of a Multiindex table (level 0, level 1) and
    returns a list of only tuples for clean multindex selection.

    Args:
        cols (list): List of strings of tuples to denote columns of a Multindex table
        df (pd.Dataframe): table with multindex columns
    """
    
    ## Create empty list for results 
    res = []
    
    ## Loop over list of column names
    for item in cols:
        ## If element is a tuple append to results
        if isinstance(item, tuple):
            res.append(item)
        ## If element is a string
        ## get all all level 1 column names and attach to results as tuples
        elif isinstance(item, str):
            res += df[[item]].columns.to_list()
            
    return(res)

def _manipulate_multicol(df, action):
    """Add and remove presuffixes to the level 0 column names
    
    Add and remove presuffixes to level 0 column names in preparation of attachment or 
    detachment of variables with same column name structure.

    Args:
        df (pd.DataFrame): Table for which the column names should be manipulated
        action (string): Action to be performed. 
            - sim: adds "sim_" presuffix
            - act: adds "act_" presuffix
            - remove: removes all presuffixes
    """
    
    ## Get complete copy of data
    data = df.copy()
    
    ## If action is "act" add act_ presuffix
    if action == "act":
        ## Get level 0 columns to add presuffix
        transform = data.columns.get_level_values(0)[
            np.invert(data.columns.get_level_values(0).isin(
                ["index","analysis"]))].to_list()
        ## Get level 0 index columns
        index = data.columns.get_level_values(0)[
            data.columns.get_level_values(0) == "index"].to_list()
        ## Get level 0 analysis columns
        analysis = data.columns.get_level_values(0)[
            data.columns.get_level_values(0) == "analysis"].to_list()
        ## Generate new level 0 column names
        new_categories = index + list(map(lambda x: "act_"+x, transform)) + analysis
        ## Get level 1 column names
        variables = data.columns.get_level_values(1)
        ## Create new Multiindex and add into table
        multi_idx = pd.MultiIndex.from_tuples(list(zip(new_categories, variables)))
        data.columns = multi_idx

    ## If action is "sim" add sim_ presuffix
    elif action == "sim":
        ## Get level 0 columns to add presuffix
        transform = data.columns.get_level_values(0)[
            np.invert(data.columns.get_level_values(0).isin(
                ["index","analysis"]))].to_list()
        ## Get level 0 index columns
        index = data.columns.get_level_values(0)[
            data.columns.get_level_values(0) == "index"].to_list()
        ## Get level 0 analysis columns
        analysis = data.columns.get_level_values(0)[
            data.columns.get_level_values(0) == "analysis"].to_list()
        ## Generate new level 0 column names
        new_categories = index + list(map(lambda x: "sim_"+x, transform)) + analysis
        ## Get level 1 column names
        variables = data.columns.get_level_values(1)
        ## Create new Multiindex and add into table
        multi_idx = pd.MultiIndex.from_tuples(list(zip(new_categories, variables)))
        data.columns = multi_idx

    ## If action is "remove" remove all presuffixes
    elif action == "remove":
        ## Remove sim_ presuffix from level 0 column names
        data.columns = data.columns.set_levels(
            data.columns.levels[0].str.replace('sim_', ''), level=0)

    return(data)

def _price_error(sim_price, act_price):
    """Calculate price simulation error

    Args:
        sim_price (np.array): Simulated price series 1D-array
        act_price (np.array): Actual price series 1D-array
    """
    
    ## Take differenc eof simulated and actual prices
    price_error = sim_price - act_price
        
    return(price_error)

def _res_demand(demand, production):
    """Calculate residual energy demand

    Args:
        demand (np.array): Electricty demand series 1D-array
        production (np.array): Energy production array N-D-array
    """
    
    ## take difference of electricity demand and sum of prodcution data
    res_demand = demand - production.sum(axis=1)/1000
     
    return(res_demand)

def _folder_check(folder):
    """Check existence of folder
    
    Checks existence of given folder path. If not existent creates it at location.

    Args:
        folder (string): Path of folder to be checked.
    """
    
    #Check existence of "graphs" directory
    isdir = os.path.isdir(folder)
    #if directory does not exist, create it
    if isdir == False:
        os.mkdir(folder)
    return()

def _gen_subtitle(time_subset=[None, None], typ_day=dict(), 
                  typ_week=dict(), type="Plot"):
    """Subtitle generation for time selections
    
    Generates a string to be used as a subtitle. String is based on time selection
    indicators supplied (see: subset_timeperiod(), typical_day(), typical_week()).

    Args:
        time_subset (list, optional): Time subset indicator. Defaults to [None, None].
        typ_day (dict, optional): Typical day selection. Defaults to dict().
        typ_week (dict, optional): Typical week selection. Defaults to dict().
        type (str, optional): Figure type to be denoted. Defaults to "Plot".
    """
    
    ## Set plot type
    subtitle = type+" for: "
    
    if any(pd.notna(time_subset)):
        time_str = ["" if pd.isna(x) else x for x in time_subset]
        time_str = []
        time_str = list(map(lambda st: str.replace(st, "-", "/"), time_str))
        subtitle += time_str[0]+" - "+time_str[1]+"; "
    
    ## If typ_day is non empty
    if bool(typ_day):
        ## Concat all items in dict together
        typ_str = list(map(lambda st: st[0]+": "+", ".join(str(i) for i in st[1]) , 
                           list(typ_day.items())))
        ## Append on base string
        typ_str = "Typical day: "+"; ".join(typ_str)
        ## Append on plot type
        subtitle += typ_str
    
    ## If typ_week is non empty    
    elif bool(typ_week):
        ## Concat all items in dict together
        typ_str = list(map(lambda st: st[0]+": "+", ".join(str(i) for i in st[1]) , 
                           list(typ_week.items())))
        ## Append on base string
        typ_str = "Typical week: "+"; ".join(typ_str)
        ## Append on plot type
        subtitle += typ_str
    
    return(subtitle)

def _Plotly_DisplaySave(fig, folder, name, show= False, save=True):
    """Setting for Plotly figure results
    
    Process settings for display and saving of a Plotly figure oject

    Args:
        fig (Plotly figure object): Plotly figure object to process.
        folder (string): Folder path to save the figure.
        name (string): File name to save figure under.
        show (bool, optional): Indicator if figure should be shown. Defaults to False.
        save (bool, optional): Indicator if figure should be saved. Defaults to True.
    """
    
    ## If indicated, save finished graph in folder
    if save:
        fig.write_html(folder+"/"+name+".html")
            
    ## If indicated, show plot, if not return nothing
    if show:
        fig.show()
        return()
    else:
        return()

def _Seaborn_DisplaySave(fig, folder, name, show= False, save=True):
    """Setting for Seaborn figure results
    
    Process settings for display and saving of a Seaborn figure oject

    Args:
        fig (Seaborn figure object): Seaborn figure object to process.
        folder (string): Folder path to save the figure.
        name (string): File name to save figure under.
        show (bool, optional): Indicator if figure should be shown. Defaults to False.
        save (bool, optional): Indicator if figure should be saved. Defaults to True.
    """
    
    ## If indicated, save finished graph in folder
    if save:
        plt.savefig(folder+"/"+name+".png")
        
    ## If indicated, show plot, if not return nothing
    if show:
        print("eval")
        plt.show()
        return()
    else:
        return()  

### Functions for loading simulated and actual data

def load_baseEoles(year, base_path=""):
    """Load output data for base Eoles model

    Load model output data, creates Multiindex and a variable description table.
    
    Args:
        year (string): Baseline year when the time data in output starts.
        base_path (str, optional): Base path where output is located. 
        Defaults to "".
    """
    ## Set access path
    if base_path == "":
        path = r"outputs\eoles_Upgrade_Outputs_hourly_generation.csv"
    else:
        path = base_path+r"\outputs\eoles_Upgrade_Outputs_hourly_generation.csv"
    
    ## Read in actual data  
    data = pd.read_csv(path, header=1, skiprows=[2,3])
    data = data.assign(hour = lambda x : pd.Timestamp(year+"-01-01") + pd.to_timedelta(x.hour, unit="H"))
    
    ## Access rows with additional information
    f = open(path, "r")
    header = f.readlines()[:4]
    f.close()
    
    ## Refine rows with additional info
    categories = _category_names(header[0])
    technologies = header[1].replace("\n", "").split(",")
    descriptions = header[2].replace("\n", "").split(",")
    units = header[3].replace("\n", "").split(",")
    
    ## Create multiIndex from categories and variable names
    idx = pd.DataFrame(list(zip_longest(*[categories, technologies], fillvalue="")))
    idx[0] = idx[0].replace("",None)
    idx[0].fillna(method="ffill", inplace=True)
    idx[0].fillna("index", inplace=True)

    ## Create Multindex from level 0 and level 1 names and assign to tables
    idx_multi = pd.MultiIndex.from_frame(idx, names=["category", "technology"])
    data.columns = idx_multi
    
    ## Create variable description table of english names, french names and variable units
    desc_df = pd.DataFrame({"desc_en": descriptions,
                            "desc_fr": None,
                            "units": units}, index=idx_multi)
    desc_df.index.names = ["categories","variables"]
    
    return(data, desc_df)
    
def load_baseHistoric(path=""):
    if path == "":
        return(pd.DataFrame())
    else:
        return()

def load_multicountryEoles(base_path=""):
    """Load output data for multicountry Eoles model

    Load model output data, creates Multiindex and a variable description table.
    Variable description currently empty, because no default description in file.

    Args:
        base_path (str, optional): Base path where output is located.. Defaults to "".
    """
    
    ## Set access paths
    if base_path == "":
        path_prices = r"outputs\prices.csv"
        path_production = r"outputs\production.csv"
    else:
        path_prices = base_path+r"\outputs\prices.csv"
        path_production = base_path+r"\outputs\production.csv"
    
    ## Load price data    
    prices = pd.read_csv(path_prices)
    ## Format datetime column
    prices["hour"] = pd.to_datetime(prices["hour"]*3600, unit="s")
    ## Unpivot data in long format
    prices = prices.melt(id_vars="hour", var_name="area", value_name="price")
    ## Create multiindex for price dataset
    price_idx = [("index","hour"),("index","area"),("cost","elec_balance_dual_values")]
    prices.columns = pd.MultiIndex.from_tuples(price_idx)  
      
    ## Load production data
    production = pd.read_csv(path_production)
    ## Format datetime column
    production["hour"] = pd.to_datetime(production["hour"]*3600, unit="s")
    
    ## Rename for consistency with actual data
    production.rename(columns={"net_exo_imports":"net_exports"}, inplace= True)
    
    ## Generate Multiindex elements level 1
    consumption = production.columns[
        production.columns.str.endswith("_in")].tolist()+["net_exports", "demand"]
    generation = production.columns[
        np.invert(production.columns.isin(consumption+["hour","area"]))].tolist()
    consumption = [item.replace("_in", "") for item in consumption]
    variables = ["area","hour"] + generation + consumption
    
    ## Gnerate Multiindex elements level 0
    categories = ["index"]*2+["generation"]*len(generation)+["consumption"]*len(consumption)
    
    ## Generate full Multiindex
    idx_multi = pd.MultiIndex.from_tuples(list(zip(categories, variables)))
    
    ## Assign Multiindex
    production.columns = idx_multi
    
    ## Merge production and prices
    data = production.merge(prices, on= [("index","hour"),("index","area")],
                            how= "inner")
    
    ## Create empty variable description dataset
    desc_df = pd.DataFrame({"desc_en": None,
                            "desc_fr": None,
                            "units": None}, index=data.columns)
    desc_df.index.names = ["categories","variables"]
    
    return(data, desc_df)

def load_multicountryHistoric(path=""):
    """Load historic data for multicountry Eoles model

    Load historic dataand creates Multiindex on table.

    Args:
        path (str, optional): Folder path where data files are located. Defaults to "".
    """
    
    ## If no path for data input is provided, return empty table
    if path == "":
        return(pd.DataFrame())
    else:
        path_prices = path+r"\actualPrices.csv"
        path_production = path+r"\actualProduction.csv"
        
    ## Load price data    
    prices = pd.read_csv(path_prices)
    ## Format datetime column
    prices["hour"] = pd.to_datetime(prices["hour"])
    ## Unpivot data in long format
    prices = prices.melt(id_vars="hour", var_name="area", value_name="price")
    ## Create multiindex for price dataset
    price_idx = [("index","hour"),("index","area"),("cost","elec_balance_dual_values")]
    prices.columns = pd.MultiIndex.from_tuples(price_idx)
       
    ## Load production data
    production = pd.read_csv(path_production)
    ## Format datetime column
    production["hour"] = pd.to_datetime(production["hour"])
    
    ## Generate Multiindex elements level 1
    consumption = production.columns[
        production.columns.str.endswith("_in")].tolist()+["net_exports"]
    generation = production.columns[
        np.invert(production.columns.isin(consumption+["hour","area"]))].tolist()
    consumption = [item.replace("_in", "") for item in consumption]
    variables = ["area","hour"] + generation + consumption
    
    ## Gnerate Multiindex elements level 0
    categories = ["index"]*2+["generation"]*len(generation)+["consumption"]*len(consumption)
    
    ## Generate full Multiindex
    idx_multi = pd.MultiIndex.from_tuples(list(zip(categories, variables)))
    
    ## Assign Multiindex
    production.columns = idx_multi
    
    ## Merge production and prices
    data = production.merge(prices, on= [("index","hour"),("index","area")],
                            how= "inner")
    
    ## Create empty variable description dataset
    #desc_df = pd.DataFrame({"desc_en": [],
    #                        "desc_fr": [],
    #                        "units": []}, index=idx_multi)
    
    return(data)

### Define refinement functions

def subset_timeperiod(df, timevar=None, time_subset=[None, None]):
    """Data subsetting for time interval
    
    Subsets a given dataframe for a indicated time iterval.

    Args:
        df (pd.DataFrame): Dataframe to subset
        timevar (string, optional): Datetime variable to use for subsetting. Defaults
        to None, and picks only datetime variable available.
        time_subset (list, optional): Time interval to subset. Defaults to [None, None]
        for full data.
    """
    
    ## Automatic timevar selector
    if pd.isna(timevar):
        timevar = df.select_dtypes(include=['datetime']).columns[0]    
    
    ## Transform time boundary to datetime
    time_subset = list(pd.to_datetime(time_subset, errors="coerce"))

    ## Replace NAs with default time boundaries
    if pd.isna(time_subset[0]):
        time_subset[0] = pd.to_datetime("1950-01-01")
    if pd.isna(time_subset[1]):
        time_subset[1] = pd.to_datetime("2199-12-31")
    
    ## Define actual boundary timestamps
    start = np.max([time_subset[0], pd.to_datetime("1950-01-01")])
    end = np.min([time_subset[1], pd.to_datetime("2199-12-12")])
    
    ## Subset dataset
    sub = df.loc[(df[timevar] >= start ) & (df[timevar] <= end),:]
    
    return(sub)

def typical_day(df, timevar=None, time_selection={"weekday":np.arange(0,7),
                                                  "month":np.arange(1,13),
                                                  "quarter": np.arange(1,5)}):
    """Computes a typical day
    
    Computes a typical day (in hours) from a full series, for an indicated time selection.

    Args:
        df (pd.DataFrame): Dataframe to subset
        timevar (string, optional): Datetime variable to use for subsetting. Defaults
        to None, and picks only datetime variable available.
        time_selection (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to {"weekday":np.arange(0,7), "month":np.arange(1,13),
        "quarter": np.arange(1,5)} which implies a typical day through the whole year.
    """
    
    ## Automatic timevar selector
    if pd.isna(timevar):
        timevar = df.select_dtypes(include=['datetime']).columns[0]
    
    ## Set default time selection
    default_params = {"weekday": np.arange(0,7),
                      "month": np.arange(1,13),
                      "quarter": np.arange(1,5)}
    
    ## Determine relevant time selectors
    selector = []
    ## Loop over selection dict
    for key, value in time_selection.items():
        ## if entry is default, not relevant, otherwise add to list
        if any(np.invert(np.isin(value, default_params[key]))):
            continue
        else:
            selector.append(key)
            
    ## Return full df if selector is empty
    if not(selector):
        return(df)
                       
    ## Create empty df for boolean selector series
    cond_df = pd.DataFrame()
    
    ##Loop over relevant selectors to create boolean series
    for select in selector:
        cond_name = "cond_"+key
        if select == "month":
            cond = (df[timevar].dt.month.isin(time_selection["month"]))
        elif select == "quarter":
            cond = (df[timevar].dt.quarter.isin(time_selection["quarter"]))
        elif select == "weekday":
            cond = (df[timevar].dt.dayofweek.isin(time_selection["weekday"]))
        ## put boolean series in selector df
        cond_df[cond_name] = cond 
    ## Aggregate boolean series up and create final for full True rows
    cond_df = cond_df.assign(cond_sum = lambda x: x.select_dtypes(include=['bool']).sum(axis=1),
                             cond_final = lambda x: x.cond_sum == len(selector))    
    
    ## Subset price data by day type
    df = df.loc[cond_df["cond_final"].to_list(),:]

    ## Extract hour indicators
    df[timevar] = df[timevar].dt.hour

    ## Aggregate price table
    summary = df.groupby(timevar).agg(["mean"])
    summary.reset_index(drop=False, inplace=True)
    summary.columns = summary.columns.droplevel(-1)
        
    return(summary)

def typical_week(df, timevar=None, time_selection={"week": np.arange(0,52),
                                                   "month":np.arange(1,13),
                                                   "quarter": np.arange(1,5)}):
    """Computes a typical week
    
    Computes a typical week (in hours) from a full series, for an indicated time selection.

    Args:
        df (pd.DataFrame): Dataframe to subset
        timevar (string, optional): Datetime variable to use for subsetting. Defaults
        to None, and picks only datetime variable available.
        time_selection (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to {"week":np.arange(0,52), "month":np.arange(1,13),
        "quarter": np.arange(1,5)} which implies a typical day through the whole year.
    """
    
    ## Automatic timevar selector
    if pd.isna(timevar):
        timevar = df.select_dtypes(include=['datetime']).columns[0]
    
    ## Set default time selection
    default_params = {"week": np.arange(0,52),
                      "month": np.arange(1,13),
                      "quarter": np.arange(1,5)}
    
    ## Determine relevant time selectors
    selector = []

    ## Loop over selection dict
    for key, value in time_selection.items():
        ## if entry is default, not relevant, otherwise add to list
        if any(np.invert(np.isin(value, default_params[key]))):
            continue
        else:
            selector.append(key)
       
    ## Return full df if selector is empty
    if not(selector):
        return(df)
           
    ## Create empty df for boolean selector series
    cond_df = pd.DataFrame()
    
    ##Loop over relevant selectors to create boolean series
    for select in selector:
        cond_name = "cond_"+key
        if select == "month":
            cond = (df[timevar].dt.month.isin(time_selection["month"]))
        elif select == "quarter":
            cond = (df[timevar].dt.quarter.isin(time_selection["quarter"]))
        elif select == "week":
            cond = (df[timevar].dt.isocalendar.week.isin(time_selection["week"]))
        ## put boolean series in selector df
        cond_df[cond_name] = cond 
    ## Aggregate boolean series up and create final for full True rows
    cond_df = cond_df.assign(cond_sum = lambda x: x.select_dtypes(include=['bool']).sum(axis=1),
                             cond_final = lambda x: x.cond_sum == len(selector))    

    ## Subset price data by day type
    df = df.loc[cond_df["cond_final"].to_list(),:]
    
    ## Extract hour indicators
    df[timevar] = df[timevar].dt.dayofweek*24 +df[timevar].dt.hour
    
    ## Aggregate price table
    summary = df.groupby(timevar).agg(["mean"])
    summary.reset_index(drop=False, inplace=True)
    summary.columns = summary.columns.droplevel(-1)
        
    return(summary)

### Define generic visualisation functions

def density_plot(data, variables, output, folder="graphs", figsize= (16,9), 
                 show=False, save=True, time_subset=[None, None], 
                 typ_day=dict(), typ_week=dict(), plot_labels={}):
    """Density plot of continuous variables
    
    Produces a density plot for one or more continuous variables.

    Args:
        data (pd.DataFrame): Table with all necessary data columns.
        variables (list): List of string variables names to visualize.
        output (string): Intended output format. "png" for seaborn graph, "html" 
        for interactive plotly.
        folder (str, optional): Folder where to save the graph results. Defaults
        to "graphs".
        figsize (tuple, optional): Proportions of figure size. Result is different
        for png and html. Defaults to (16,9).
        show (bool, optional): Indicator if figure should be shown. Defaults to False.
        save (bool, optional): Indicator if figure should be saved. Defaults to True.
        time_subset (list, optional): Time interval to subset. Defaults to [None, None]
        for full data.
        typ_day (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to dict(), implying no selection.
        typ_week (dict, optional):  Indicates selection for the calculation of a
        typical week. Defaults to dict(), implying no selection.
        plot_labels (dict, optional): Dictionary of plot label elements. Defaults to {}, implying a standard
        plot type file name and variable name legend labels.
            - title: title of plot
            - x_axis: x-axis label
            - y_axis: y-axis label
            - legend_title: title of Legend section
            - cat_labels: labels of elements in legend
            - file_name: name for plot file to be saved
    """
    
    ## Set default labels
    default_labels = {"title":"",
                      "x_axis":"",
                      "y_axis":"",
                      "legend_title":"",
                      "cat_labels":"",
                      "file_name":"density_plot"}
    
    ## Process label indications
    for label in default_labels.keys():
        if np.invert(label in plot_labels.keys()):
            plot_labels[label] = default_labels[label]
    
    ## Transform input "variables" into list if not one
    if not(isinstance(variables, list)):
        variables = [variables]
    
    ## Check inputs for typical functions
    assert not(bool(typ_day) & bool(typ_week)), "Please give only one of typ_day and typ_week."

    ## Check for existence of save folder
    _folder_check(folder)
    
    ## Automatic timevar selector
    timevar = data.select_dtypes(include=['datetime']).columns[0]
    
    ## Apply refinement selection functions
    data = subset_timeperiod(data, timevar = timevar, time_subset=time_subset)
    data = typical_day(data, timevar = timevar, time_selection=typ_day)
    data = typical_week(data, timevar = timevar, time_selection=typ_week)
    
    ## Set cat_labels to variable names if not indicated
    if (plot_labels["cat_labels"] == "") & isinstance(data.columns, pd.MultiIndex):
        plot_labels["cat_labels"] = list(map(lambda x: x[1], variables))
    elif plot_labels["cat_labels"] == "":
        plot_labels["cat_labels"] = variables
    
    ## If any of selection filled, generate a subtitle
    if bool(typ_day) | bool(typ_week) | any(pd.notna(time_subset)):
        subtitle = _gen_subtitle(time_subset=time_subset, typ_day=typ_day,
                                 typ_week=typ_week)
    else:
        subtitle= ""

    ## Generate plot based on output type
    if output == "png": #Use matplotlib/seaborn for static plots
        ## Set basic plotting parameters
        fig = plt.figure(figsize=figsize)
        sns.set_style('whitegrid')
        ## Generate plot itself
        for var in variables:
            sns.kdeplot(data[var])#, color="r")
        ## Add plot labels
        plt.title(subtitle, loc="left")
        plt.suptitle(plot_labels["title"], y=0.93, x=0.2)
        plt.ylabel(plot_labels["y_axis"])
        plt.xlabel(plot_labels["x_axis"])
        plt.legend(labels= plot_labels["cat_labels"],
                   title = plot_labels["legend_title"], loc=2)
        ## Process Save and Display parameters
        _Seaborn_DisplaySave(fig=fig, name= plot_labels["file_name"],
                             folder = folder, show= show, save= save)
        
    elif output == "html": #Use plotly for dynamic plots
        ## Initialize figure object
        fig = go.Figure()
        ## Create actual plot
        data_list = list(map(lambda x: data[x], variables))
        fig = ff.create_distplot(data_list, 
                                 group_labels=plot_labels["cat_labels"], 
                                 show_hist=False, show_rug=False)
        ## Manipulate aesthetics
        title = plot_labels["title"]+"<br><sup>"+subtitle+"</sup>"
        fig.update_layout(title={'text': title,'xanchor': "auto"},
                          xaxis_title= plot_labels["x_axis"], 
                          yaxis_title= plot_labels["y_axis"],
                          legend=dict(yanchor="top", y=0.99,
                                      xanchor="left", x=0.01),
                          legend_title= dict(text= plot_labels["legend_title"]), 
                          margin=dict(t=70,b=0,l=0,r=0),
                          width=figsize[0]*100, height=figsize[1]*100)
        ## Process Save and Display parameters
        _Plotly_DisplaySave(fig=fig, name= plot_labels["file_name"], 
                            folder = folder, show= show, save= save)
    return()

def cdf_plot(data, variables, output, folder="graphs", figsize= (16,9), 
             show=False, save=True, time_subset=[None, None], typ_day=dict(),
             typ_week=dict(), plot_labels={}):
    """Cumulative distribution plot of continuous variables
    
    Produces a cumulative distribution plot for one or more continuous variables.

    Args:
        data (pd.DataFrame): Table with all necessary data columns.
        variables (list): List of string variables names to visualize.
        output (string): Intended output format. "png" for seaborn graph, "html" 
        for interactive plotly.
        folder (str, optional): Folder where to save the graph results. Defaults
        to "graphs".
        figsize (tuple, optional): Proportions of figure size. Result is different
        for png and html. Defaults to (16,9).
        show (bool, optional): Indicator if figure should be shown. Defaults to False.
        save (bool, optional): Indicator if figure should be saved. Defaults to True.
        time_subset (list, optional): Time interval to subset. Defaults to [None, None]
        for full data.
        typ_day (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to dict(), implying no selection.
        typ_week (dict, optional):  Indicates selection for the calculation of a
        typical week. Defaults to dict(), implying no selection.
        plot_labels (dict, optional): Dictionary of plot label elements. Defaults to {}, implying a standard
        plot type file name and variable name legend labels.
            - title: title of plot
            - x_axis: x-axis label
            - y_axis: y-axis label
            - legend_title: title of Legend section
            - cat_labels: labels of elements in legend
            - file_name: name for plot file to be saved
    """
    
    ## Set default labels
    default_labels = {"title":"",
                      "x_axis":"",
                      "y_axis":"",
                      "legend_title":"",
                      "cat_labels":"",
                      "file_name":"cdf_plot"}
    
    ## Process label indications
    for label in default_labels.keys():
        if np.invert(label in plot_labels.keys()):
            plot_labels[label] = default_labels[label]
    
    ## Transform input "variables" into list if not one
    if not(isinstance(variables, list)):
        variables = [variables]
    
    ## Check inputs for typical functions
    assert not(bool(typ_day) & bool(typ_week)), "Please give only one of typ_day and typ_week."

    ## Check for existence of save folder
    _folder_check(folder)
    
    ## Automatic timevar selector
    timevar = data.select_dtypes(include=['datetime']).columns[0]
    
    ## Apply refinement selection functions
    data = subset_timeperiod(data, timevar = timevar, time_subset=time_subset)
    data = typical_day(data, timevar = timevar, time_selection=typ_day)
    data = typical_week(data, timevar = timevar, time_selection=typ_week)
    data.pop(timevar)
    
    ## Get rid of Multindex for plotting and set aside technology names
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel()
        variables = list(map(lambda x: x[1], variables))
    ## Pivot data into appropriate shape    
    data = data.melt(value_vars= variables, var_name='source', value_name='value')
            
    ## Set cat_labels to variable names if not indicated
    if (plot_labels["cat_labels"] == "") & isinstance(data.columns, pd.MultiIndex):
        plot_labels["cat_labels"] = list(map(lambda x: x[1], variables))
    elif plot_labels["cat_labels"] == "":
        plot_labels["cat_labels"] = variables
    
    ## If any of selection filled, generate a subtitle
    if bool(typ_day) | bool(typ_week) | any(pd.notna(time_subset)):
        subtitle = _gen_subtitle(time_subset=time_subset, typ_day=typ_day,
                                 typ_week=typ_week)
    else:
        subtitle= ""

    ## Generate plot based on output type
    if output == "png": #Use matplotlib/seaborn for static plots
        ## Set basic plotting parameters
        fig = plt.figure(figsize=figsize)
        sns.set_style('whitegrid')
        
        ## Generate plot itself
        sns.ecdfplot(data, x= "value", hue="source")

        ## Add plot labels
        plt.title(subtitle, loc="left")
        plt.suptitle(plot_labels["title"], y=0.93, x=0.2)
        plt.xlabel(plot_labels["x_axis"])
        plt.legend(labels= plot_labels["cat_labels"], 
                   title = plot_labels["legend_title"], loc=2)
        ## Process Save and Display parameters
        _Seaborn_DisplaySave(fig=fig, name= plot_labels["file_name"],
                             folder = folder, show= show, save= save)
        
    elif output == "html": #Use plotly for dynamic plots
        ## Initialize figure object
        fig = go.Figure()
        ## Create actual plot
        fig = px.ecdf(data, x="value", color= "source")
            #labels=dict(zip(list(data.columns), cat_labels)))
        ## Manipulate aesthetics
        title = plot_labels["title"]+"<br><sup>"+subtitle+"</sup>"
        fig.update_layout(title={'text': title,'xanchor': 'auto'},
                          xaxis_title= plot_labels["x_axis"],
                          yaxis_title= plot_labels["y_axis"],
                          legend=dict(yanchor="top", y=0.99,
                                      xanchor="left", x=0.01),
                          legend_title= dict(text= plot_labels["legend_title"]), 
                          margin=dict(t=70,b=0,l=0,r=0),
                          width=figsize[0]*100, height=figsize[1]*100)
        ## Process Save and Display parameters
        _Plotly_DisplaySave(fig=fig, name= plot_labels["file_name"],
                            folder = folder, show= show, save= save)
    return()

def price_duration_curve(data, prices, output, folder="graphs", figsize= (16,9), 
                         show=False, save=True, time_subset=[None, None], 
                         typ_day=dict(), typ_week=dict(), plot_labels={}):
    """Price duration curve plot of price variables
    
    Produces a price duration curve for one or more price variables.

    Args:
        data (pd.DataFrame): Table with all necessary data columns.
        prices (list): List of string price variable names to visualize.
        output (string): Intended output format. "png" for seaborn graph, "html" 
        for interactive plotly.
        folder (str, optional): Folder where to save the graph results. Defaults
        to "graphs".
        figsize (tuple, optional): Proportions of figure size. Result is different
        for png and html. Defaults to (16,9).
        show (bool, optional): Indicator if figure should be shown. Defaults to False.
        save (bool, optional): Indicator if figure should be saved. Defaults to True.
        time_subset (list, optional): Time interval to subset. Defaults to [None, None]
        for full data.
        typ_day (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to dict(), implying no selection.
        typ_week (dict, optional):  Indicates selection for the calculation of a
        typical week. Defaults to dict(), implying no selection.
        plot_labels (dict, optional): Dictionary of plot label elements. Defaults to {}, implying a standard
        plot type file name and variable name legend labels.
            - title: title of plot
            - x_axis: x-axis label
            - y_axis: y-axis label
            - legend_title: title of Legend section
            - cat_labels: labels of elements in legend
            - file_name: name for plot file to be saved
    """
    
    ## Set default labels
    default_labels = {"title":"",
                      "x_axis":"",
                      "y_axis":"",
                      "legend_title":"",
                      "cat_labels":"",
                      "file_name":"price_duration_curve"}
    
    ## Process label indications
    for label in default_labels.keys():
        if np.invert(label in plot_labels.keys()):
            plot_labels[label] = default_labels[label]
    
    ## Transform input "variables" into list if not one
    if not(isinstance(prices, list)):
        prices = [prices]
    
    ## Check inputs for typical functions
    assert not(bool(typ_day) & bool(typ_week)), "Please give only one of typ_day and typ_week."

    ## Check for existence of save folder
    _folder_check(folder)
    
    ## Automatic timevar selector
    timevar = data.select_dtypes(include=['datetime']).columns[0]
    
    ## Apply refinement selection functions
    data = subset_timeperiod(data, timevar = timevar, time_subset=time_subset)
    data = typical_day(data, timevar = timevar, time_selection=typ_day)
    data = typical_week(data, timevar = timevar, time_selection=typ_week)
    
    ## Set cat_labels to variable names if not indicated
    if (plot_labels["cat_labels"] == "") & isinstance(data.columns, pd.MultiIndex):
        plot_labels["cat_labels"] = list(map(lambda x: x[1], prices))
    elif plot_labels["cat_labels"] == "":
        plot_labels["cat_labels"] = prices
    
    ## If any of selection filled, generate a subtitle
    if bool(typ_day) | bool(typ_week) | any(pd.notna(time_subset)):
        subtitle = _gen_subtitle(time_subset=time_subset, typ_day=typ_day,
                                 typ_week=typ_week)
    else:
        subtitle= ""
            
    ## Get rid of Multindex for plotting and set aside technology names
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel()
        prices = list(map(lambda x: x[1], prices))
        
    ## Generate quantile data
    plot_df = pd.DataFrame(np.arange(0.05,0.95,0.05), columns=["pct"])
    plot_df[prices] = data[prices].apply(
        lambda x: np.nanquantile(x, 1-plot_df["pct"]), axis=0)

    ## Pivot data into appropriate shape    
    plot_df = plot_df.melt(id_vars= ["pct"], value_vars= prices, var_name='source',
                           value_name='value')
      
    ## Generate plot based on output type
    if output == "png": #Use matplotlib/seaborn for static plots
        ## Set basic plotting parameters
        fig = plt.figure(figsize=figsize)
        sns.set_style('whitegrid')
        ## Generate plot itself
        sns.lineplot(data= plot_df, x= "pct", y= "value", hue= "source")
        ## Add plot labels
        plt.title(subtitle, loc="left")
        plt.suptitle(plot_labels["title"], y=0.93, x=0.2)
        plt.xlabel(plot_labels["x_axis"])
        plt.ylabel(plot_labels["y_axis"])
        plt.legend(labels= plot_labels["cat_labels"], 
                   title = plot_labels["legend_title"], loc=1)
        ## Process Save and Display parameters
        _Seaborn_DisplaySave(fig=fig, name= plot_labels["file_name"],
                             folder = folder, show= show, save= save)
        
    elif output == "html": #Use plotly for dynamic plots
        ## Initialize figure object
        fig = go.Figure()
        ## Create actual plot
        cat_labels = pd.DataFrame(zip(prices, plot_labels["cat_labels"]), 
                                  columns=["source","labels"])
        plot_df = plot_df.merge(cat_labels, on="source",how="left")
        fig = px.line(plot_df, x="pct", y="value", color="labels")
        ## Manipulate aesthetics
        title = plot_labels["title"]+"<br><sup>"+subtitle+"</sup>"
        fig.update_layout(title={'text': title,'xanchor': 'auto'},
                          xaxis_title= plot_labels["x_axis"],
                          yaxis_title= plot_labels["y_axis"],
                          legend=dict(yanchor="top", y=0.99,
                                      xanchor="right", x=0.99),
                          legend_title= dict(text= plot_labels["legend_title"]), 
                          margin=dict(t=70,b=0,l=0,r=0),
                          width=figsize[0]*100, height=figsize[1]*100)
        ## Process Save and Display parameters
        _Plotly_DisplaySave(fig=fig, name= plot_labels["file_name"],
                            folder = folder, show= show, save= save)
    return()

def stacked_areachart(data, variables, output, timevar=None, add_line=None, 
                      folder="graphs", figsize= (16,9), show=False, save=True,
                      time_subset=[None, None], typ_day=dict(), typ_week=dict(),
                      plot_labels={}):
    """Stacked area chart of continuous variables
    
    Produces a stacked area chart for one or more continuous variables.

    Args:
        data (pd.DataFrame): Table with all necessary data columns.
        variables (list): List of string variables names to visualize.
        output (string): Intended output format. "png" for seaborn graph, "html" 
        for interactive plotly.
        timevar (str, optional): Variable to use for time axis. Defaults to None, 
        when first datetime variable is picked.
        add_line (str, optional): Variable to use for display of a lineplot over 
        the areachart. Defaults to None.
        folder (str, optional): Folder where to save the graph results. Defaults
        to "graphs".
        figsize (tuple, optional): Proportions of figure size. Result is different
        for png and html. Defaults to (16,9).
        show (bool, optional): Indicator if figure should be shown. Defaults to False.
        save (bool, optional): Indicator if figure should be saved. Defaults to True.
        time_subset (list, optional): Time interval to subset. Defaults to [None, None]
        for full data.
        typ_day (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to dict(), implying no selection.
        typ_week (dict, optional):  Indicates selection for the calculation of a
        typical week. Defaults to dict(), implying no selection.
        plot_labels (dict, optional): Dictionary of plot label elements. Defaults to {}, implying a standard
        plot type file name and variable name legend labels.
            - title: title of plot
            - x_axis: x-axis label
            - y_axis: y-axis label
            - legend_title: title of Legend section
            - cat_labels: labels of elements in legend
            - file_name: name for plot file to be saved
    """
    
    ## Set default labels
    default_labels = {"title":"",
                      "x_axis":"",
                      "y_axis":"",
                      "legend_title":"",
                      "cat_labels": variables,
                      "file_name":"stacked_areachart"}
    ## Set default legend labels
    if isinstance(data.columns, pd.MultiIndex):
        default_labels["cat_labels"] = list(map(lambda x: x[1], variables))
    if pd.notna(add_line):
        default_labels["cat_labels"] += add_line
    
    ## Process label indications
    for label in default_labels.keys():
        if np.invert(label in plot_labels.keys()):
            plot_labels[label] = default_labels[label]
    
    ## Transform input "variables" into list if not one
    if not(isinstance(variables, list)):
        variables = [variables]
    
    ## Check inputs for typical functions
    assert not(bool(typ_day) & bool(typ_week)), "Please give only one of typ_day and typ_week."

    ## Check for existence of save folder
    _folder_check(folder)
    
    ## Automatic timevar selector
    if pd.isna(timevar):
        timevar = data.select_dtypes(include=['datetime']).columns[0]
        
    ## Apply refinement selection functions
    data = subset_timeperiod(data, timevar = timevar, time_subset=time_subset)
    data = typical_day(data, timevar = timevar, time_selection=typ_day)
    data = typical_week(data, timevar = timevar, time_selection=typ_week)
    
    ## Set cat_labels to variable names if not indicated
    if (plot_labels["cat_labels"] == "") & isinstance(data.columns, pd.MultiIndex):
        plot_labels["cat_labels"] = list(map(lambda x: x[1], variables))
    elif plot_labels["cat_labels"] == "":
        plot_labels["cat_labels"] = variables
    
    ## If any of selection filled, generate a subtitle
    if bool(typ_day) | bool(typ_week) | any(pd.notna(time_subset)):
        subtitle = _gen_subtitle(time_subset=time_subset, typ_day=typ_day,
                                 typ_week=typ_week)
    else:
        subtitle= ""

    ## Generate plot based on output type
    if output == "png": #Use matplotlib/seaborn for static plots
        ## Set basic plotting parameters
        fig = plt.figure(figsize=figsize)
        sns.set_style('whitegrid')
        ## Generate plot itself
        fig = plt.stackplot(data[timevar].T, data[variables].T)
        
        ## Add additional lineplot if wanted
        if pd.notna(add_line): 
            fig = sns.lineplot(x=data[timevar], y=data[add_line], 
                               palette='black', linewidth=2.5)
            
        ## Add plot labels
        plt.title(subtitle, loc="left")
        plt.suptitle(plot_labels["title"], y=0.93, x=0.2)
        plt.ylabel(plot_labels["y_axis"])
        plt.xlabel(plot_labels["x_axis"])
        if len(variables) > 1:
            plt.legend(labels= plot_labels["cat_labels"], 
                       title = plot_labels["legend_title"], loc=2)
        else:
            plt.legend([],[], frameon=False)
        ## Process Save and Display parameters
        _Seaborn_DisplaySave(fig=fig, name= plot_labels["file_name"],
                             folder = folder, show= show, save= save)
        
    elif output == "html": #Use plotly for dynamic plots
        ## Initialize figure object
        fig = go.Figure()
        ## Create actual plot
        for idx, var in enumerate(variables):
            fig.add_trace(go.Scatter(x = data[timevar], y = data[var],
                                     name = plot_labels["cat_labels"][idx],
                                     stackgroup='one'))
        if pd.notna(add_line):
            fig.add_trace(go.Scatter(x = data[timevar], y = data[add_line], mode="lines",
                                     name= plot_labels["cat_labels"][-1]))
                
        ## Manipulate aesthetics
        title = plot_labels["title"]+"<br><sup>"+subtitle+"</sup>"
        fig.update_layout(title={'text': title,'xanchor': 'auto'},
                          xaxis_title= plot_labels["x_axis"], yaxis_title= plot_labels["y_axis"],
                          legend=dict(yanchor="top", y=0.99,
                                      xanchor="left", x=0.01),
                          legend_title= dict(text= plot_labels["legend_title"]), 
                          margin=dict(t=70,b=0,l=0,r=0),
                          width=figsize[0]*100, height=figsize[1]*100)
        if len(variables) == 1:
            fig.update_layout(showlegend=False)
        ## Process Save and Display parameters
        _Plotly_DisplaySave(fig=fig, name= plot_labels["file_name"],
                            folder = folder, show= show, save= save)
    
    return()

def stacked_barplot(data, x, variables, output, display= "relative", folder="graphs",
                    figsize= (16,9), show=False,save=True, time_subset=[None, None],
                    typ_day=dict(), typ_week=dict(), plot_labels={}):
    """Stacked barplot of continuous variables over discrete categories
    
    Produces a stacked barplot for continuous variables over a discrete variable.

    Args:
        data (pd.DataFrame): Table with all necessary data columns.
        x (string): String variable name to use for x axis display.
        variables (list): List of string variables names to visualize.
        output (string): Intended output format. "png" for seaborn graph, "html" 
        for interactive plotly.
        display (str, optional): Indicator for whether plot should display absolute 
        or relative proprortions. Defaults to "relative".        
        folder (str, optional): Folder where to save the graph results. Defaults
        to "graphs".
        figsize (tuple, optional): Proportions of figure size. Result is different
        for png and html. Defaults to (16,9).
        show (bool, optional): Indicator if figure should be shown. Defaults to False.
        save (bool, optional): Indicator if figure should be saved. Defaults to True.
        time_subset (list, optional): Time interval to subset. Defaults to [None, None]
        for full data.
        typ_day (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to dict(), implying no selection.
        typ_week (dict, optional):  Indicates selection for the calculation of a
        typical week. Defaults to dict(), implying no selection.
        plot_labels (dict, optional): Dictionary of plot label elements. Defaults to {}, implying a standard
        plot type file name and variable name legend labels.
            - title: title of plot
            - x_axis: x-axis label
            - y_axis: y-axis label
            - legend_title: title of Legend section
            - cat_labels: labels of elements in legend
            - file_name: name for plot file to be saved
    """

    ## Set default labels
    default_labels = {"title":"",
                      "x_axis":"",
                      "y_axis":"",
                      "legend_title":"",
                      "cat_labels":"",
                      "file_name":"stacked_barplot"}
    
    ## Process label indications
    for label in default_labels.keys():
        if np.invert(label in plot_labels.keys()):
            plot_labels[label] = default_labels[label]
    
    ## Transform input "variables" into list if not one
    if not(isinstance(variables, list)):
        variables = [variables]
    
    ## Check inputs for typical functions
    assert not(bool(typ_day) & bool(typ_week)), "Please give only one of typ_day and typ_week."

    ## Check for existence of save folder
    _folder_check(folder)

    ## Automatic timevar selector
    timevar = data.select_dtypes(include=['datetime']).columns[0]
    
    ## Apply refinement selection functions
    data = subset_timeperiod(data, timevar = timevar, time_subset=time_subset)
    data = typical_day(data, timevar = timevar, time_selection=typ_day)
    data = typical_week(data, timevar = timevar, time_selection=typ_week)
    ## Discard timevar variable
    data.pop(timevar)
    
    ## Aggregate dataset 
    data = data[variables].agg(["sum"])   
    data = data.T
    data.reset_index(inplace=True, drop=False)

    ## Select if abolute number of relative number should be displayed
    if display == "relative": # Relative to total of category
        data = data.merge(data[[x,"sum"]].groupby([x]).sum().reset_index(),
                      on=x, suffixes=("","_total"))
        data = data.assign(share = lambda x: (x["sum"]/x["sum_total"])*100)
        data = data.iloc[:,[0,1,4]]
        data.columns = ["category","grouping","weight"]
        #data.rename(columns={"share": "weight"}, inplace=True)

    elif display == "absolute": # Absolute
        data.columns = ["category","grouping","weight"]

    ## Set cat_labels to variable names if not indicated
    if plot_labels["cat_labels"] == "":
        plot_labels["cat_labels"] = data["grouping"]
    
    ## Generate text for title and axes
    data["cat_labels"] = plot_labels["cat_labels"]
    # if any of selection filled, generate a subtitle
    if bool(typ_day) | bool(typ_week) | any(pd.notna(time_subset)):
        subtitle = _gen_subtitle(time_subset=time_subset, typ_day=typ_day,
                                 typ_week=typ_week)
    else:
        subtitle= ""
      
    ## Generate plot based on output type
    if output == "png": #Use matplotlib/seaborn for static plots
        ## Set basic plotting parameters
        fig = plt.figure(figsize=figsize)
        sns.set_style('whitegrid')
        ## Generate plot itself
        a = sns.histplot(data, x='category', hue='grouping', weights='weight',
                         multiple='stack', shrink=0.5, stat= "count", linewidth=0.1,
                         edgecolor='black')
        sns.move_legend(a, loc="upper left", bbox_to_anchor=(1, 1), 
                        title='Legend')
        ## Add plot labels
        plt.title(subtitle, loc="left")
        plt.suptitle(plot_labels["title"], y=0.93, x=0.2)
        plt.xlabel(plot_labels["x_axis"])
        plt.xlabel(plot_labels["y_axis"])

        ## Process Save and Display parameters
        _Seaborn_DisplaySave(fig=fig, name= plot_labels["file_name"],
                             folder = folder, show= show, save= save)
        
    elif output == "html": #Use plotly for dynamic plots
        ## Initialize figure object
        fig = go.Figure()
        ## Create actual plot
        fig = px.bar(data, x="category", y="weight", color="grouping")
        fig.update_traces(width=0.5)
        ## Manipulate aesthetics
        title = plot_labels["title"]+"<br><sup>"+subtitle+"</sup>"
        fig.update_layout(title={'text': title,'xanchor': 'auto'},
                          xaxis_title= plot_labels["x_axis"], 
                          yaxis_title= plot_labels["y_axis"],
                          legend=dict(yanchor="top", y=0.99,
                                      xanchor="right", x=0.99),
                          legend_title= dict(text= plot_labels["legend_title"]), 
                          margin=dict(t=70,b=0,l=0,r=0),
                          width=figsize[0]*100, height=figsize[1]*100)
        ## Process Save and Display parameters
        _Plotly_DisplaySave(fig=fig, name= plot_labels["file_name"],
                            folder = folder, show= show, save= save)
    
    return()

def energy_piechart(data, variables, output, folder="graphs", figsize= (16,9),
                    show=False, save=True, time_subset=[None, None], typ_day=dict(),
                    typ_week=dict(), plot_labels={}):
    """Pie chart over several variables
    
    Produces a pie chart of several variables in proportion to the table total.

    Args:
        data (pd.DataFrame): Table with all necessary data columns.
        variables (list): List of string variables names to visualize.
        output (string): Intended output format. "png" for seaborn graph, "html" 
        for interactive plotly.
        folder (str, optional): Folder where to save the graph results. Defaults
        to "graphs".
        figsize (tuple, optional): Proportions of figure size. Result is different
        for png and html. Defaults to (16,9).
        show (bool, optional): Indicator if figure should be shown. Defaults to False.
        save (bool, optional): Indicator if figure should be saved. Defaults to True.
        time_subset (list, optional): Time interval to subset. Defaults to [None, None]
        for full data.
        typ_day (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to dict(), implying no selection.
        typ_week (dict, optional):  Indicates selection for the calculation of a
        typical week. Defaults to dict(), implying no selection.
        plot_labels (dict, optional): Dictionary of plot label elements. Defaults to {}, implying a standard
        plot type file name and variable name legend labels.
            - title: title of plot
            - legend_title: title of Legend section
            - cat_labels: labels of elements in legend
            - file_name: name for plot file to be saved
    """
    
    ## Set default labels
    default_labels = {"title":"",
                      "legend_title":"",
                      "cat_labels":"",
                      "file_name":"energy_piechart"}
    
    ## Process label indications
    for label in default_labels.keys():
        if np.invert(label in plot_labels.keys()):
            plot_labels[label] = default_labels[label]
    
    ## Transform input "variables" into list if not one
    if not(isinstance(variables, list)):
        variables = [variables]
    
    ## Check inputs for typical functions
    assert not(bool(typ_day) & bool(typ_week)), "Please give only one of typ_day and typ_week."

    ## Check for existence of save folder
    _folder_check(folder)

    ## Automatic timevar selector
    timevar = data.select_dtypes(include=['datetime']).columns[0]
    
    ## Apply refinement selection functions
    data = subset_timeperiod(data, timevar = timevar, time_subset=time_subset)
    data = typical_day(data, timevar = timevar, time_selection=typ_day)
    data = typical_week(data, timevar = timevar, time_selection=typ_week)
    ## Discard timevar variable
    data.pop(timevar)
    
    ## Aggregate data for pie chart
    data = data[variables].agg(["sum"])
    data.columns = data.columns.droplevel()
    data = data.T
    data.reset_index(inplace=True, drop=False)
    
    ## Set cat_labels to variable names if not indicated
    if plot_labels["cat_labels"] == "":
        plot_labels["cat_labels"] = data["technology"]
    
    ## Generate text for title and axes
    data["cat_labels"] = plot_labels["cat_labels"]
    # if any of selection filled, generate a subtitle
    if bool(typ_day) | bool(typ_week) | any(pd.notna(time_subset)):
        subtitle = _gen_subtitle(time_subset=time_subset, typ_day=typ_day,
                                 typ_week=typ_week)
    else:
        subtitle= ""
      
    ## Generate plot based on output type
    if output == "png": #Use matplotlib/seaborn for static plots
        ## Set basic plotting parameters
        fig = plt.figure(figsize=figsize)
        sns.set_style('whitegrid')
        ## Generate plot itself
        plt.pie(x=data["sum"], labels=data["cat_labels"], autopct="%1.2f%%")
        ## Add plot labels
        plt.title(subtitle, loc="left")
        plt.suptitle(plot_labels["title"], y=0.93, x=0.2)
        #plt.xlabel(x_axis)
        plt.legend(labels= data["cat_labels"], title = plot_labels["legend_title"], loc=2)
        ## Process Save and Display parameters
        _Seaborn_DisplaySave(fig=fig, name= plot_labels["file_name"],
                             folder = folder, show= show, save= save)
        
    elif output == "html": #Use plotly for dynamic plots
        ## Initialize figure object
        fig = go.Figure()
        ## Create actual plot
        fig = px.pie(data, values="sum", names='cat_labels')
        ## Manipulate aesthetics
        title = plot_labels["title"]+"<br><sup>"+subtitle+"</sup>"
        fig.update_layout(title={'text': title,'xanchor': 'auto'},
                          legend=dict(yanchor="top", y=0.99,
                                      xanchor="right", x=0.99),
                          legend_title= dict(text= plot_labels["legend_title"]), 
                          margin=dict(t=70,b=0,l=0,r=0),
                          width=figsize[0]*100, height=figsize[1]*100)
        ## Process Save and Display parameters
        _Plotly_DisplaySave(fig=fig, name= plot_labels["file_name"],
                            folder = folder, show= show, save= save)
    return()

def energy_line(data, variables, output, timevar=None, folder="graphs", 
                figsize= (16,9), show=False, save=True, time_subset=[None, None],
                typ_day=dict(), typ_week=dict(), plot_labels={}):
    """Line plot of continuous variables
    
    Produces a line plot for one or more continuous variables over a time dimension.

    Args:
        data (pd.DataFrame): Table with all necessary data columns.
        variables (list): List of string variables names to visualize.
        output (string): Intended output format. "png" for seaborn graph, "html" 
        for interactive plotly.
        timevar (str, optional): Variable to use for time axis. Defaults to None, 
        when first datetime variable is picked.
        folder (str, optional): Folder where to save the graph results. Defaults
        to "graphs".
        figsize (tuple, optional): Proportions of figure size. Result is different
        for png and html. Defaults to (16,9).
        show (bool, optional): Indicator if figure should be shown. Defaults to False.
        save (bool, optional): Indicator if figure should be saved. Defaults to True.
        time_subset (list, optional): Time interval to subset. Defaults to [None, None]
        for full data.
        typ_day (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to dict(), implying no selection.
        typ_week (dict, optional):  Indicates selection for the calculation of a
        typical week. Defaults to dict(), implying no selection.
        plot_labels (dict, optional): Dictionary of plot label elements. Defaults to {}, implying a standard
        plot type file name and variable name legend labels.
            - title: title of plot
            - x_axis: x-axis label
            - y_axis: y-axis label
            - legend_title: title of Legend section
            - cat_labels: labels of elements in legend
            - file_name: name for plot file to be saved
    """
    
    ## Set default labels
    default_labels = {"title":"",
                      "x_axis":"",
                      "y_axis":"",
                      "legend_title":"",
                      "cat_labels":"",
                      "file_name":"lineplot"}
    
    ## Process label indications
    for label in default_labels.keys():
        if np.invert(label in plot_labels.keys()):
            plot_labels[label] = default_labels[label]
    
    ## Transform input "variables" into list if not one
    if not(isinstance(variables, list)):
        variables = [variables]
    
    ## Check inputs for typical functions
    assert not(bool(typ_day) & bool(typ_week)), "Please give only one of typ_day and typ_week."

    ## Check for existence of save folder
    _folder_check(folder)
    
    ## Automatic timevar selector
    if pd.isna(timevar):
        timevar = data.select_dtypes(include=['datetime']).columns[0]
    
    ## Subset data for relevant variables
    data = data[variables+[timevar]]

    ## Apply refinement selection functions
    data = subset_timeperiod(data, timevar = timevar, time_subset=time_subset)
    data = typical_day(data, timevar = timevar, time_selection=typ_day)
    data = typical_week(data, timevar = timevar, time_selection=typ_week)

    ## Set cat_labels to variable names if not indicated
    if (plot_labels["cat_labels"] == "") & isinstance(data.columns, pd.MultiIndex):
        plot_labels["cat_labels"] = list(map(lambda x: x[1], variables))
    elif plot_labels["cat_labels"] == "":
        plot_labels["cat_labels"] = variables

    ## If any of selection filled, generate a subtitle
    if bool(typ_day) | bool(typ_week) | any(pd.notna(time_subset)):
        subtitle = _gen_subtitle(time_subset=time_subset, typ_day=typ_day,
                                 typ_week=typ_week)
    else:
        subtitle= ""

    ## Get rid of Multindex for plotting and set aside technology names
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel()
        variables = list(map(lambda x: x[1], variables))
        timevar = "hour"
    ## Pivot data into appropriate shape    
    data = data.melt(id_vars= [timevar], value_vars= data.columns[
        data.columns != timevar].to_list(), var_name='source', value_name='value')

    ## Generate plot based on output type
    if output == "png": #Use matplotlib/seaborn for static plots
        ## Set basic plotting parameters
        fig = plt.figure(figsize=figsize)
        sns.set_style('whitegrid')
        ## Generate plot itself
        sns.lineplot(data= data, x= timevar, y= "value", hue= "source")
        ### add geom_smooth equivalent
        
        ## Add plot labels
        plt.title(subtitle, loc="left")
        plt.suptitle(plot_labels["title"], y=0.93, x=0.2)
        plt.ylabel(plot_labels["y_axis"])
        plt.xlabel(plot_labels["x_axis"])
        if len(variables) > 1:
            plt.legend(labels= plot_labels["cat_labels"], 
                       title = plot_labels["legend_title"], loc=2)
        else:
            plt.legend([],[], frameon=False)
        ## Process Save and Display parameters
        _Seaborn_DisplaySave(fig=fig, name= plot_labels["file_name"],
                             folder = folder, show= show, save= save)
        
    elif output == "html": #Use plotly for dynamic plots
        ## Initialize figure object
        fig = go.Figure()
        ## Create actual plot
        if not(plot_labels["cat_labels"]):
            fig = px.line(data, x=timevar, y="value", color="source")#, labels=)
        else:
            cat_labels = pd.DataFrame(zip(variables, plot_labels["cat_labels"]), 
                                      columns=["source","labels"])
            data = data.merge(cat_labels, on="source",how="left")
            fig = px.line(data, x=timevar, y="value", color="labels")
        
        ## Manipulate aesthetics
        title = plot_labels["title"]+"<br><sup>"+subtitle+"</sup>"
        fig.update_layout(title={'text': title,'xanchor': 'auto'},
                          xaxis_title= plot_labels["x_axis"], 
                          yaxis_title= plot_labels["y_axis"],
                          legend=dict(yanchor="top", y=0.99,
                                      xanchor="left", x=0.01),
                          legend_title= dict(text= plot_labels["legend_title"]), 
                          margin=dict(t=70,b=0,l=0,r=0),
                          width=figsize[0]*100, height=figsize[1]*100)
        if len(variables) == 1:
            fig.update_layout(showlegend=False)
        ## Process Save and Display parameters
        _Plotly_DisplaySave(fig=fig, name= plot_labels["file_name"],
                            folder = folder, show= show, save= save)
    return()

def bivar_scatter(data, x, y, output, folder="graphs", figsize= (16,9),
                  show=False, save=True, time_subset=[None, None],
                  typ_day=dict(), typ_week=dict(), plot_labels={}):
    """Bivariate scatter plot of continuous variables
    
    Produces a bivariate scatter plot for one or more continuous variables.

    Args:
        data (pd.DataFrame): Table with all necessary data columns.
        x (string): Name of variable to graph on x-axis.
        y (string): Name of variable to graph on y-axis.
        output (string): Intended output format. "png" for seaborn graph, "html" 
        for interactive plotly.
        folder (str, optional): Folder where to save the graph results. Defaults
        to "graphs".
        figsize (tuple, optional): Proportions of figure size. Result is different
        for png and html. Defaults to (16,9).
        show (bool, optional): Indicator if figure should be shown. Defaults to False.
        save (bool, optional): Indicator if figure should be saved. Defaults to True.
        time_subset (list, optional): Time interval to subset. Defaults to [None, None]
        for full data.
        typ_day (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to dict(), implying no selection.
        typ_week (dict, optional):  Indicates selection for the calculation of a
        typical week. Defaults to dict(), implying no selection.
        plot_labels (dict, optional): Dictionary of plot label elements. Defaults to {}, implying a standard
        plot type file name and variable name legend labels.
            - title: title of plot
            - x_axis: x-axis label
            - y_axis: y-axis label
            - legend_title: title of Legend section
            - cat_labels: labels of elements in legend
            - file_name: name for plot file to be saved
    """
    
    ## Set default labels
    default_labels = {"title":"",
                      "x_axis":"",
                      "y_axis":"",
                      "file_name":"bivar_scatter"}
    
    ## Process label indications
    for label in default_labels.keys():
        if np.invert(label in plot_labels.keys()):
            plot_labels[label] = default_labels[label]
    
    ## Check inputs for typical functions
    assert not(bool(typ_day) & bool(typ_week)), "Please give only one of typ_day and typ_week."

    ## Check for existence of save folder
    _folder_check(folder)
    
    ## Automatic timevar selector
    timevar = data.select_dtypes(include=['datetime']).columns[0]
    
    ## Apply refinement selection functions
    data = subset_timeperiod(data, timevar = timevar, time_subset=time_subset)
    data = typical_day(data, timevar = timevar, time_selection=typ_day)
    data = typical_week(data, timevar = timevar, time_selection=typ_week)
    ## Discard time variable
    data.pop(timevar)
    
    ## If any of selection filled, generate a subtitle
    if bool(typ_day) | bool(typ_week) | any(pd.notna(time_subset)):
        subtitle = _gen_subtitle(time_subset=time_subset, typ_day=typ_day,
                                 typ_week=typ_week)
    else:
        subtitle= "" 
          
    ## Generate plot based on output type
    if output == "png": #Use matplotlib/seaborn for static plots
        ## Set basic plotting parameters
        fig = plt.figure(figsize=figsize)
        sns.set_style('whitegrid')
        ## Generate plot itself
        sns.scatterplot(x=data[x],y=data[y], color= "black")
        
        ## Add plot labels
        plt.title(subtitle, loc="left")
        plt.suptitle(plot_labels["title"], y=0.93, x=0.2)
        plt.ylabel(plot_labels["y_axis"])
        plt.xlabel(plot_labels["x_axis"])
        
        ## Process Save and Display parameters
        _Seaborn_DisplaySave(fig=fig, name= plot_labels["file_name"],
                             folder = folder, show= show, save= save)
        
    elif output == "html": #Use plotly for dynamic plots
        ## Initialize figure object
        fig = go.Figure()
        ## Create actual plot
        fig = px.scatter(x=data[x],y=data[y], color_discrete_sequence=["black"])
        #fig.add_trace(go.Scatter(x=data[x],y=data[y], name="spline",line_shape='spline'))
                                 #text=["tweak line smoothness<br>with 'smoothing' in line object"],
                    #hoverinfo='text+name'))
                    
        ## Manipulate aesthetics
        title = plot_labels["title"]+"<br><sup>"+subtitle+"</sup>"
        fig.update_layout(title={'text': title,'xanchor': 'auto'},
                          xaxis_title= plot_labels["x_axis"], 
                          yaxis_title= plot_labels["y_axis"],
                          legend=dict(yanchor="top", y=0.99,
                                      xanchor="left", x=0.01),
                          #legend_title= dict(text= leg_title), 
                          margin=dict(t=70,b=0,l=0,r=0),
                          width=figsize[0]*100, height=figsize[1]*100)
        ## Process Save and Display parameters
        _Plotly_DisplaySave(fig=fig, name= plot_labels["file_name"],
                            folder = folder, show= show, save= save)
    return()

def reg_energydata(data, dep, indep, squared=[], logged=[], constant=True,
                   include_aov=True, time_subset=[None, None],
                   typ_day=dict(), typ_week=dict()):
    """Multivariate regression

    Produce a multivariate regression output with options for functional form transformations and
    AOV display.

    Args:
        data (pd.DataFrame): Data table to use in regression.
        dep (string): Variable name of dependent variable.
        indep (list): List of variable names of independent variables.
        squared (list, optional): List of variable names to  square transform. Defaults to [].
        logged (list, optional): List of variable names to log transform. Defaults to [].
        constant (bool, optional): Indicator on whether to include a constant in regression output.
        Defaults to True.
        include_aov (bool, optional): Indicator on whether to include an analysis of variance section 
        in regression output. Defaults to True.
        time_subset (list, optional): Time interval to subset. Defaults to [None, None]
        for full data.
        typ_day (dict, optional): Indicates selection for the calculation of a
        typical day. Defaults to dict(), implying no selection.
        typ_week (dict, optional):  Indicates selection for the calculation of a
        typical week. Defaults to dict(), implying no selection.
    """
    
    ## Transform input "indep" into list if not one
    if not(isinstance(indep, list)):
        indep = [indep]
    
    ## Check inputs for typical functions
    assert not(bool(typ_day) & bool(typ_week)), "Please give only one of typ_day and typ_week."
            
    ## Automatic timevar selector
    if pd.isna(timevar):
        timevar = data.select_dtypes(include=['datetime']).columns[0]
    
    ## Subset for indicated time period
    data = subset_timeperiod(data, timevar = timevar, time_subset=time_subset)
    data = typical_day(data, timevar = timevar, time_selection=typ_day)
    data = typical_week(data, timevar = timevar, time_selection=typ_week)
    ## Discard timevar variable    
    data.pop(timevar)
    
    ##Apply functional form transformations: squared
    if not(squared):
        squared_names = [var+"_sq" for var in squared]
        data[squared_names] = data[squared].apply(lambda x: x**2, axis=1)
    else:
        squared_names = []
    ##Apply functional form transformations: logarithm
    if not(logged):
        logged_names = [var+"_log" for var in logged]
        data[logged_names] = data[logged].apply(lambda x: np.log(x), axis=1)
    else:
        logged_names = []
    
    ## Specify linear regression model components
    Y = data[dep] #dependent variable
    X = data[indep+squared_names+logged_names] #independent variables
    if constant: #constant if wanted
        X = sm.add_constant(X)
    
    ## Estimate regression model
    if bool(typ_day) | bool(typ_week) | any(pd.notna(time_subset)):
        title = _gen_subtitle(time_subset=time_subset, typ_day=typ_day, 
                              yp_week=typ_week, type= "Regression")
        print(title)
    model = sm.OLS(Y,X)
    results_lm = model.fit()
    print(results_lm.summary())
    
    ## Estimate analysis of variance
    if include_aov:
        aov_table = sm.stats.anova_lm(model, typ= 2, robust="hc3")
        print(aov_table)
    
    return()

### Eoles Base model visualisation object

class Eoles_baseline:
    
    def __init__(self, year, base_path="", historic_path=""):
        
        self.data, self.vardesc_df = load_baseEoles(year= year, base_path= base_path)
        
        self.historic = load_baseHistoric(path= historic_path) 

    def attach_historic(self, historic_path=""):
        
        ## If historic is not loaded and no path given return message
        if all([self.historic.empty, not(historic_path)]):
            return("No historic file or file path given.")
        ## If historic is not loaded, perform loading
        elif all([(historic_path != ""), self.historic.empty]):
            self.historic = load_baseHistoric(path=historic_path)
        
        ## Get vardesc dataframe to update
        cols = self.data.columns[np.invert(
            self.data.columns.get_level_values(0).isin(["index","analysis"]))
            ].to_list()
        
        ## Manipulate sim data vardesc 
        simdesc_df = self.vardesc_df.loc[cols,:]
        simdesc_df.reset_index(drop=False, inplace=True)
        simdesc_df = simdesc_df.assign(categories = lambda x: ("sim_"+x.categories),
                                       desc_en = lambda x: "simulated "+x.desc_en)
        simdesc_df.set_index(["categories","variables"],drop=True, inplace=True)
        
        ## Manipulate act data vardesc 
        actdesc_df = self.vardesc_df.loc[cols,:]
        actdesc_df.reset_index(drop=False, inplace=True)
        actdesc_df = actdesc_df.assign(categories = lambda x: ("act_"+x.categories),
                                       desc_en = lambda x: "actual "+x.desc_en)
        actdesc_df.set_index(["categories","variables"],drop=True, inplace=True)
        
        ## Concat vardesc dfs together
        if "analysis" in self.data.columns.levels[0]:
            restdesc_df = self.vardesc_df.loc[["index","analysis"],:]
            vardesc_df = pd.concat([restdesc_df, simdesc_df, actdesc_df], axis=0)
        else:
            restdesc_df = self.vardesc_df.loc["index",:]
            vardesc_df = pd.concat([restdesc_df, simdesc_df, actdesc_df], axis=0)
        
        self.vardesc_df = vardesc_df.copy()
        
        ## Get simulated data
        sim_data = self.data.copy()
        ## Add sim presuffix
        sim_data = _manipulate_multicol(sim_data, action="sim")
        
        ## Get actual data
        act_data = self.historic.copy()
        ## Add act presuffix
        act_data = _manipulate_multicol(act_data, action="act")
        
        ## Merge simulated and actual data and reassign to object
        merge_data = sim_data.merge(act_data, on=[("index","hour")], how="inner")
        self.data = merge_data.copy() 
                 
    def detach_historic(self):
        
        ## Get columns of simulated variables
        cols = self.data.columns.levels[0][
            self.data.columns.levels[0].str.startswith("sim")].to_list()
        cols = np.unique(cols).tolist()
        if "analysis" in self.data.columns.levels[0]:
            cols = ["index"]+cols+["analysis"]
        else:
            cols = ["index"]+cols
        
        ## Subset data in object
        data = self.data[cols]
        data.columns = data.columns.remove_unused_levels()
        ## Remove sim presuffix
        data = _manipulate_multicol(data, action="remove")
        ## Reassign data to object
        self.data = data.copy()
        
        ## Get vardesc of simulated variables
        vardesc = self.vardesc_df.copy()
              
    def update_vardesc(self, changes=pd.DataFrame()):
            
        vardesc_new = self.vardesc_df.join(changes, how="left", rsuffix= "_new")
            
        for var in changes.columns.to_list():
            
            vardesc_new[var].fillna(vardesc_new.loc[:,var+"_new"],
                                        inplace=True)
            vardesc_new.pop(var+"_new")
        
        self.vardesc_df = vardesc_new.copy()  

    def compute_analysisvar(self, variables):
        
        ## Attach historic file
        self.attach_historic()
        
        ## Ensure only valid variables are requested
        valid_entries = ["res_demand","diffres_demand",
                         "price_error","absprice_error"]
        variables = [var for var in variables if var in valid_entries]
        
        data = self.data.copy()
        
        for var in variables:
            if var == "res_demand":
                data[("analysis","res_demand")] = _res_demand(
                    data[("sim_consumption", "electricity_demand")],
                    data[[]])
            elif var == "diffres_demand":
                data[("analysis","diffres_demand")] = _res_demand(
                    data[("sim_consumption", "electricity_demand")],
                    data[[]]).diff()
                
            elif var == "price_error":
                data[("analysis","price_error")] = _price_error(
                    sim_price=data[("sim_cost", "electricity_demand")],
                    act_price=data[("sim_cost", "electricity_demand")])
                
            elif var == "absprice_error":
                data[("analysis","absprice_error")] = abs(_price_error(
                    sim_price=data[("sim_cost", "electricity_demand")],
                    act_price=data[("sim_cost", "electricity_demand")]))
                
        self.data = data.copy()
        
        ## Detach historic file
        self.detach_historic()
        
        ana_vars = list(zip(["analysis"]*len(valid_entries), valid_entries))
        desc_en = ["electricity demand residual after renewable generation",
                   "lagged electricity demand residual after renewable generation",
                   "difference between simulated and actual energy price",
                   "absolute difference between simulated and actual energy price"]
        desc_fr = None
        units = ["GWh-e","GWh-e","Euros/MWe","Euros/MWe"]
        ana_vardesc = pd.DataFrame({"desc_en": desc_en,
                                    "desc_fr": desc_fr,
                                    "units": units}, 
                                   index=ana_vars)
        ana_vars_wanted = list(zip(["analysis"]*len(variables), variables))
        ana_vardesc = ana_vardesc.loc[ana_vars_wanted,:]
        self.vardesc_df = pd.concat([self.vardesc_df, ana_vardesc], axis=0)
        
    def density_plot(self, variables, output, folder="graphs", figsize= (16,9),
                     show=False, save=True, time_subset=[None, None], 
                     typ_day=dict(), typ_week=dict(), label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(variables, list)):
            variables = [variables]
            
        ## Flatten column index over multiindex
        variables = _index_flatselect(variables, self.data)
        
        ## Generate plotting data
        data = self.data[variables+[("index","hour")]]
        
        ## Generate text for title and axes
        if len(variables) == 1:
            title = "Line chart of "+self.vardesc_df.loc[variables,label_lang].tolist()[0]
            y_axis = self.vardesc_df.loc[variables,label_lang].tolist()[0]
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "linechart_"+variables[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
        else:
            title = "Multiline chart"
            y_axis = "Values"
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "multilinechart-"+"-".join(list(map(lambda x: x[1], variables)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
 
        ## Call plotting function
        density_plot(data= data, variables=variables, output=output, folder=folder,
                     figsize=figsize, show=show, save=save, time_subset=time_subset,
                     typ_day=typ_day, typ_week=typ_week, 
                     plot_labels={
                         "title": title,
                         "x_axis": x_axis,
                         "y_axis": y_axis,
                         "legend_title": leg_title,
                         "cat_labels": cat_labels,
                         "file_name": file_name
                         })
        return()
        
    def cdf_plot(self, variables, output, folder="graphs", figsize= (16,9),
                 show=False, save=True, time_subset=[None, None], 
                 typ_day=dict(), typ_week=dict(), label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(variables, list)):
            variables = [variables]
            
        ## Flatten column index over multiindex
        variables = _index_flatselect(variables, self.data)
        
        ## Generate plotting data
        data = self.data[variables+[("index","hour")]]
        
        ## Generate text for title and axes
        if len(variables) == 1:
            title = "Line chart of "+self.vardesc_df.loc[variables,label_lang].tolist()[0]
            y_axis = self.vardesc_df.loc[variables,label_lang].tolist()[0]
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "linechart_"+variables[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
        else:
            title = "Multiline chart"
            y_axis = "Values"
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "multilinechart-"+"-".join(list(map(lambda x: x[1], variables)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
 
        ## Call plotting function
        cdf_plot(data= data, variables=variables, output=output, folder=folder,
                 figsize=figsize, show=show, save=save, time_subset=time_subset,
                 typ_day=typ_day, typ_week=typ_week, 
                 plot_labels={
                     "title": title,
                     "x_axis": x_axis,
                     "y_axis": y_axis,
                     "legend_title": leg_title,
                     "cat_labels": cat_labels,
                     "file_name": file_name
                     })
        return()
             
    def price_duration_curve(self, prices, output, folder="graphs", figsize= (16,9),
                             show=False, save=True, time_subset=[None, None], 
                             typ_day=dict(), typ_week=dict(), label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(prices, list)):
            prices = [prices]
            
        ## Flatten column index over multiindex
        prices = _index_flatselect(prices, self.data)
        
        ## Generate plotting data
        data = self.data[prices+[("index","hour")]]
        
        ## Generate text for title and axes
        if len(prices) == 1:
            title = "Line chart of "+self.vardesc_df.loc[prices,label_lang].tolist()[0]
            y_axis = self.vardesc_df.loc[prices,label_lang].tolist()[0]
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "linechart_"+prices[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[prices,label_lang].tolist()
        else:
            title = "Multiline chart"
            y_axis = "Values"
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "multilinechart-"+"-".join(list(map(lambda x: x[1], prices)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[prices,label_lang].tolist()
 
        ## Call plotting function
        price_duration_curve(data= data, prices=prices, output=output, 
                             folder=folder, figsize=figsize, show=show, save=save,
                             time_subset=time_subset, typ_day=typ_day, 
                             typ_week=typ_week, plot_labels={
                                 "title": title,
                                 "x_axis": x_axis,
                                 "y_axis": y_axis,
                                 "legend_title": leg_title,
                                 "cat_labels": cat_labels,
                                 "file_name": file_name
                                 })
        return()
    
    def stacked_areachart(self, variables, output, negative=[], add_line=None, 
                          folder="graphs", figsize= (16,9), show=False, save=True,
                          time_subset=[None, None], typ_day=dict(), typ_week=dict(),
                          label_lang="desc_en"):

        ## Transform input "variables" into list if not one
        if not(isinstance(variables, list)):
            variables = [variables]
            
        ## Flatten column index over multiindex
        variables = _index_flatselect(variables, self.data)

        ## Get data
        if pd.notna(add_line):
            data = self.data[variables+[add_line]+[("index","hour")]]
        else:
            data = self.data[variables+[("index","hour")]]

        ## Apply negative transformation if intended
        if negative != []:
            data[negative] = data[negative]*(-1)
        
        ## Generate text for title and axes
        if len(variables) == 1:
            title = "Area chart of "+self.vardesc_df.loc[variables,label_lang].tolist()[0]
            y_axis = self.vardesc_df.loc[variables,label_lang].tolist()[0]
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "areachart_"+variables[0][1]
            #cat_labels = [self.vardesc_df.loc[variables,label_lang].tolist()[0]]
        else:
            title = "Stacked Area Chart"
            y_axis = "Values"
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "areachart-"+"-".join(list(map(lambda x: x[1], variables)))
            leg_title = "Legend"
            if pd.isna(add_line):
                cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
            else:
                cat_labels = self.vardesc_df.loc[variables+[add_line],
                                                 label_lang].tolist()
                
        
        ## Call plotting function
        stacked_areachart(data=data, stack_variables=variables, output=output,
                          add_line=add_line, folder=folder, figsize=figsize, show=show,
                          save=save, time_subset=time_subset, typ_day=typ_day, 
                          typ_week=typ_week, plot_labels={
                              "title": title,
                              "x_axis": x_axis,
                              "y_axis": y_axis,
                              "legend_title": leg_title,
                              "cat_labels": cat_labels,
                              "file_name": file_name
                          })
        
        return()
        
    def bivar_scatter(self, x, y, output, folder="graphs", figsize= (16,9),
                      show=False, save=True, time_subset=[None, None],
                      typ_day=dict(), typ_week=dict(), label_lang="desc_en"):
        
        ## Get plotting data
        data = self.data[[x,y]+[("index","hour")]]
        
        ## Generate text for title and axes
        title = "Scatterplot of "+self.vardesc_df.loc[x,label_lang].tolist()[0]+" and "+self.vardesc_df.loc[y,label_lang].tolist()[0]
        y_axis = self.vardesc_df.loc[y,label_lang].tolist()[0]
        x_axis = self.vardesc_df.loc[x,label_lang].tolist()[0]
        file_name = "scatter_"+self.vardesc_df.loc[x,label_lang].tolist()[0]+"_"+self.vardesc_df.loc[y,label_lang].tolist()[0]
        
        ## Call plotting function
        bivar_scatter(data= data, x=x, y=y, output=output,folder=folder, figsize=figsize,
                      show=show,save=save, time_subset=time_subset, typ_day=typ_day,
                      typ_week=typ_week, plot_labels={
                              "title": title,
                              "x_axis": x_axis,
                              "y_axis": y_axis,
                              "file_name": file_name
                          })
        
        return()

    def energy_line(self, variables, output, folder="graphs", figsize= (16,9),
                    show=False, save=True, time_subset=[None, None], typ_day=dict(),
                    typ_week=dict(), label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(variables, list)):
            variables = [variables]
            
        ## Flatten column index over multiindex
        variables = _index_flatselect(variables, self.data)
        
        ## Generate plotting data
        data = self.data[variables+[("index","hour")]]
        
        ## Generate text for title and axes
        if len(variables) == 1:
            title = "Line chart of "+self.vardesc_df.loc[variables,label_lang].tolist()[0]
            y_axis = self.vardesc_df.loc[variables,label_lang].tolist()[0]
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "linechart_"+variables[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
        else:
            title = "Multiline chart"
            y_axis = "Values"
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "multilinechart-"+"-".join(list(map(lambda x: x[1], variables)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
 
        ## Call plotting function
        energy_line(data= data, variables=variables, output=output, folder=folder,
                    figsize=figsize, show=show, save=save, time_subset=time_subset,
                    typ_day=typ_day, typ_week=typ_week, 
                    plot_labels={
                        "title": title,
                        "x_axis": x_axis,
                        "y_axis": y_axis,
                        "legend_title": leg_title,
                        "cat_labels": cat_labels,
                        "file_name": file_name
                        })
        
        return()

    def energy_piechart(self, variables, output, folder="graphs", figsize= (16,9),
                        show=False,save=True, time_subset=[None, None], typ_day=dict(),
                        typ_week=dict(), label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(variables, list)):
            variables = [variables]
            
        ## Flatten column index over multiindex
        variables = _index_flatselect(variables, self.data)
        
        ## Generate plotting data
        data = self.data[variables+[("index","hour")]]
        
        ## Generate text for title and axes
        if len(variables) == 1:
            title = "Pie chart of "+self.vardesc_df.loc[variables,label_lang].tolist()[0]
            file_name = "piechart_"+variables[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
        else:
            title = "Pie chart"
            file_name = "piechart-"+"-".join(list(map(lambda x: x[1], variables)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
 
        ## Call plotting function
        energy_piechart(data= data, variables=variables, output=output, folder=folder,
                        figsize=figsize, show=show, save=save, time_subset=time_subset,
                        typ_day=typ_day, typ_week=typ_week, 
                        plot_labels={
                            "title": title,
                            "legend_title": leg_title,
                            "cat_labels": cat_labels,
                            "file_name": file_name
                            })
        
        return()
    
    def reg_energydata(self, dep, indep, squared=[], logged=[], constant=True,
                       include_aov=True, time_subset=[None, None],
                       typ_day=dict(), typ_week=dict()):#, label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(indep, list)):
            indep = [indep]
            
        ## Flatten column index over multiindex
        indep = _index_flatselect(indep, self.data)
        
        ## Generate plotting data
        data = self.data[indep+[("index","hour")]]
        
        ## Call regression function
        reg_energydata(data=data, dep=dep, indep=indep, squared=squared, 
                       logged=logged, constant=constant, include_aov=include_aov,
                       time_subset=time_subset, typ_day=typ_day, typ_week=typ_week)
        
        return()

    def energy_balancechart(self, display, output, folder="graphs", figsize= (16,9),
                            show=False, save=True, time_subset=[None, None],
                            typ_day=dict(), typ_week=dict(), label_lang= "desc_en"):

        ## Flatten column index over multiindex
        variables = _index_flatselect(["generation","consumption"], self.data)

        ## Get plotting data
        data = self.data[variables+[("index","hour")]]
        
        ## Generate text for title and axes
        if display == "absolute":
            y_axis = "Energy constitution in GWh"
        elif display == "relative":
            y_axis = "Energy constitution in %"
        title = "Energy balance chart"
        x_axis = ""
        file_name = "energy_balancechart"
        leg_title = "Legend"
        cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
 
        ## Call plotting function
        stacked_barplot(data= data, x="category", stack_variables="variables", 
                        output=output, display=display , folder=folder, figsize=figsize,
                        show=show, save=save, time_subset=time_subset, typ_day=typ_day,
                        typ_week=typ_week, plot_labels={
                            "title": title,
                            "y_axis": y_axis,
                            "x_axis": x_axis,
                            "legend_title": leg_title,
                            "cat_labels": cat_labels,
                            "file_name": file_name
                            })
                
        return()

### Eoles Multicountry model visualisation object

class Eoles_multicountry:
    
    def __init__(self, base_path, historic_path=""):
        
        self.data, self.vardesc_df = load_multicountryEoles(base_path= base_path)
        
        self.historic = load_multicountryHistoric(path= historic_path)
        
    def attach_historic(self, historic_path=""):
        
        ## If historic is not loaded and no path given return message
        if all([self.historic.empty, not(historic_path)]):
            return("No historic file or file path given.")
        ## If historic is not loaded, perform loading
        elif all([(historic_path != ""), self.historic.empty]):
            self.historic = load_multicountryHistoric(path=historic_path)
        
        ## Get vardesc dataframe to update
        cols = self.data.columns[np.invert(
            self.data.columns.get_level_values(0).isin(["index","analysis"]))
            ].to_list()
        
        ## Manipulate sim data vardesc 
        simdesc_df = self.vardesc_df.loc[cols,:]
        simdesc_df.reset_index(drop=False, inplace=True)
        simdesc_df = simdesc_df.assign(categories = lambda x: ("sim_"+x.categories),
                                       desc_en = lambda x: "simulated "+x.desc_en)
        simdesc_df.set_index(["categories","variables"],drop=True, inplace=True)

        ## Manipulate act data vardesc 
        actdesc_df = self.vardesc_df.loc[cols,:]
        actdesc_df.reset_index(drop=False, inplace=True)
        actdesc_df = actdesc_df.assign(categories = lambda x: "act_"+x.categories,
                                       desc_en = lambda x: "actual "+x.desc_en)
        actdesc_df.set_index(["categories","variables"],drop=True, inplace=True)

        ## Concat vardesc dfs together
        if "analysis" in self.data.columns.levels[0]:
            restdesc_df = self.vardesc_df.loc[["index","analysis"],:]
            vardesc_df = pd.concat([restdesc_df, simdesc_df, actdesc_df], axis=0)
        else:
            restdesc_df = self.vardesc_df.loc[["index"],:]
            vardesc_df = pd.concat([restdesc_df, simdesc_df, actdesc_df], axis=0)

        self.vardesc_df = vardesc_df.copy()
        
        ## Get simulated data
        sim_data = self.data.copy()
        ## Add sim presuffix
        sim_data = _manipulate_multicol(sim_data, action="sim")
        
        ## Get actual data
        act_data = self.historic.copy()
        ## Add act presuffix
        act_data = _manipulate_multicol(act_data, action="act")
        
        ## Merge simulated and actual data and reassign to object
        merge_data = sim_data.merge(act_data, on=[("index","hour"),("index","area")],
                                    how="inner")
        self.data = merge_data.copy()
                 
    def detach_historic(self):
        
        ## Get vardesc of simulated variables
        cols = self.data.columns.levels[0][np.invert(
            self.data.columns.levels[0].isin(["index","analysis"]))]
        cols = cols[cols.str.startswith("sim")].to_list()
        
        ## Manipulate sim data vardesc 
        simdesc_df = self.vardesc_df.loc[cols,:]
        simdesc_df.reset_index(drop=False, inplace=True)
        simdesc_df = simdesc_df.assign(categories = lambda x: x.categories.str.replace("sim_",""),
                                       desc_en = lambda x: x.desc_en.str.replace("simulated ",""))
        simdesc_df.set_index(["categories","variables"],drop=True, inplace=True)
        
        ## Concat vardesc dfs together
        if "analysis" in self.data.columns.levels[0]:
            restdesc_df = self.vardesc_df.loc[["index","analysis"],:]
            vardesc_df = pd.concat([restdesc_df, simdesc_df], axis=0)
        else:
            restdesc_df = self.vardesc_df.loc[["index"],:]
            vardesc_df = pd.concat([restdesc_df, simdesc_df], axis=0)
            
        self.vardesc_df = vardesc_df.copy()
        
        ## Get columns of simulated variables
        cols = self.data.columns.levels[0][
            self.data.columns.levels[0].str.startswith("sim")].to_list()
        cols = np.unique(cols).tolist()
        if "analysis" in self.data.columns.levels[0]:
            cols = ["index"]+cols+["analysis"]
        else:
            cols = ["index"]+cols

        ## Subset data in object
        data = self.data[cols]
        data.columns = data.columns.remove_unused_levels()

        ## Remove sim presuffix
        data = _manipulate_multicol(data, action="remove")
        ## Reassign data to object
        self.data = data.copy()

    def update_vardesc(self, changes=pd.DataFrame()):
        
        vardesc_new = self.vardesc_df.join(changes, how="left", 
                                           rsuffix= "_new")
            
        for var in changes.columns.to_list():
            
            vardesc_new[var].fillna(vardesc_new.loc[:,var+"_new"],
                                        inplace=True)
            vardesc_new.pop(var+"_new")
        
        self.vardesc_df = vardesc_new.copy()
                
    def compute_analysisvar(self, variables):
        
        ## Attach historic file
        self.attach_historic()
        
        ## Ensure only valid variables are requested
        valid_entries = ["res_demand","diffres_demand",
                         "price_error","absprice_error",
                         "net_IM"]
        variables = [var for var in variables if var in valid_entries]
        
        data = self.data.copy()
        
        for var in variables:
            if var == "res_demand":
                data[("analysis","res_demand")] = _res_demand(
                    data[("sim_consumption", "demand")],
                    data[[("act_generation", "pv"),
                          ("act_generation", "river"),
                          ("act_generation", "wind")]])
            elif var == "diffres_demand":
                data[("analysis","diffres_demand")] = _res_demand(
                    data[("sim_consumption", "demand")],
                    data[[("act_generation", "pv"),
                          ("act_generation", "river"),
                          ("act_generation", "wind")]]).diff()
                
            elif var == "price_error":
                data[("analysis","price_error")] = _price_error(
                    sim_price=data[("sim_cost", "elec_balance_dual_values")],
                    act_price=data[("act_cost", "elec_balance_dual_values")])
                
            elif var == "absprice_error":
                data[("analysis","absprice_error")] = abs(_price_error(
                    sim_price=data[("sim_cost", "elec_balance_dual_values")],
                    act_price=data[("act_cost", "elec_balance_dual_values")]))
            elif var == "net_IM":
                data[("analysis","net_IM")] = np.sum(data[("act_generation","net_imports"), 
                                                          ("act_consumption","net_exports")],
                                                     axis= 0)
                
        self.data = data.copy()
        
        ## Detach historic file
        self.detach_historic()
        
        ana_vars = list(zip(["analysis"]*len(valid_entries), valid_entries))
        desc_en = ["electricity demand residual after renewable generation",
                   "lagged electricity demand residual after renewable generation",
                   "difference between simulated and actual energy price",
                   "absolute difference between simulated and actual energy price",
                   "Net difference between energy imports and exports"]
        desc_fr = None
        units = ["GWh-e","GWh-e","Euros/MWe","Euros/MWe","GWh-e"]
        ana_vardesc = pd.DataFrame({"desc_en": desc_en,
                                    "desc_fr": desc_fr,
                                    "units": units}, 
                                   index=ana_vars)
        ana_vars_wanted = list(zip(["analysis"]*len(variables), variables))
        ana_vardesc = ana_vardesc.loc[ana_vars_wanted,:]
        self.vardesc_df = pd.concat([self.vardesc_df, ana_vardesc], axis=0)
     
    def density_plot(self, variables, country, output, folder="graphs", figsize= (16,9),
                     show=False, save=True, time_subset=[None, None], 
                     typ_day=dict(), typ_week=dict(), label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(variables, list)):
            variables = [variables]
            
        ## Flatten column index over multiindex
        variables = _index_flatselect(variables, self.data)
        
        ## Generate plotting data
        data = self.data[variables+[("index","hour"), ("index","area")]]
        data = data.loc[data[("index","area")] == country,:]
        data.pop(("index","area"))
        
        ## Generate text for title and axes
        if len(variables) == 1:
            title = "Line chart of "+self.vardesc_df.loc[variables,label_lang].tolist()[0]
            y_axis = self.vardesc_df.loc[variables,label_lang].tolist()[0]
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "linechart_"+variables[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
        else:
            title = "Multiline chart"
            y_axis = "Values"
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "multilinechart-"+"-".join(list(map(lambda x: x[1], variables)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
 
        ## Call plotting function
        density_plot(data= data, variables=variables, output=output, folder=folder,
                     figsize=figsize, show=show, save=save, time_subset=time_subset,
                     typ_day=typ_day, typ_week=typ_week, 
                     plot_labels={
                         "title": title,
                         "x_axis": x_axis,
                         "y_axis": y_axis,
                         "legend_title": leg_title,
                         "cat_labels": cat_labels,
                         "file_name": file_name
                         })
        return()
        
    def cdf_plot(self, variables, country, output, folder="graphs", figsize= (16,9),
                 show=False, save=True, time_subset=[None, None], 
                 typ_day=dict(), typ_week=dict(), label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(variables, list)):
            variables = [variables]
            
        ## Flatten column index over multiindex
        variables = _index_flatselect(variables, self.data)
        
        ## Generate plotting data
        data = self.data[variables+[("index","hour"), ("index","area")]]
        data = data.loc[data[("index","area")] == country,:]
        data.pop(("index","area"))
        
        ## Generate text for title and axes
        if len(variables) == 1:
            title = "Line chart of "+self.vardesc_df.loc[variables,label_lang].tolist()[0]
            y_axis = self.vardesc_df.loc[variables,label_lang].tolist()[0]
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "linechart_"+variables[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
        else:
            title = "Multiline chart"
            y_axis = "Values"
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "multilinechart-"+"-".join(list(map(lambda x: x[1], variables)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
 
        ## Call plotting function
        cdf_plot(data= data, variables=variables, output=output, folder=folder,
                 figsize=figsize, show=show, save=save, time_subset=time_subset,
                 typ_day=typ_day, typ_week=typ_week, 
                 plot_labels={
                     "title": title,
                     "x_axis": x_axis,
                     "y_axis": y_axis,
                     "legend_title": leg_title,
                     "cat_labels": cat_labels,
                     "file_name": file_name
                     })
        return()
             
    def price_duration_curve(self, prices, country, output, folder="graphs", figsize= (16,9),
                             show=False, save=True, time_subset=[None, None], 
                             typ_day=dict(), typ_week=dict(), label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(prices, list)):
            prices = [prices]
            
        ## Flatten column index over multiindex
        prices = _index_flatselect(prices, self.data)
        
        ## Generate plotting data
        data = self.data[prices+[("index","hour"), ("index","area")]]
        data = data.loc[data[("index","area")] == country,:]
        data.pop(("index","area"))
        
        ## Generate text for title and axes
        if len(prices) == 1:
            title = "Line chart of "+self.vardesc_df.loc[prices,label_lang].tolist()[0]
            y_axis = self.vardesc_df.loc[prices,label_lang].tolist()[0]
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "linechart_"+prices[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[prices,label_lang].tolist()
        else:
            title = "Multiline chart"
            y_axis = "Values"
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "multilinechart-"+"-".join(list(map(lambda x: x[1], prices)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[prices,label_lang].tolist()
 
        ## Call plotting function
        price_duration_curve(data= data, prices=prices, output=output, 
                             folder=folder, figsize=figsize, show=show, save=save,
                             time_subset=time_subset, typ_day=typ_day, 
                             typ_week=typ_week, plot_labels={
                                 "title": title,
                                 "x_axis": x_axis,
                                 "y_axis": y_axis,
                                 "legend_title": leg_title,
                                 "cat_labels": cat_labels,
                                 "file_name": file_name
                                 })
        return()
    
    def energy_line(self, variables, country, output, folder="graphs", figsize= (16,9),
                    show=False, save=True, time_subset=[None, None], typ_day=dict(),
                    typ_week=dict(), label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(variables, list)):
            variables = [variables]
            
        ## Flatten column index over multiindex
        variables = _index_flatselect(variables, self.data)
        
        ## Generate plotting data
        data = self.data[variables+[("index","hour"), ("index","area")]]
        data = data.loc[data[("index","area")] == country,:]
        data.pop(("index","area"))
        
        ## Generate text for title and axes
        if len(variables) == 1:
            var = self.vardesc_df.loc[variables,label_lang].tolist()[0]
            title = "Line chart of "+var+" in "+country
            y_axis = var
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "linechart_"+country+"_"+variables[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
        else:
            title = "Multiline chart for "+country
            y_axis = "Values"
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "multilinechart_"+country+"_".join(list(map(lambda x: x[1], variables)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
 
        ## Call plotting function
        energy_line(data= data, variables=variables, output=output, folder=folder,
                    figsize=figsize, show=show, save=save, time_subset=time_subset,
                    typ_day=typ_day, typ_week=typ_week, 
                    plot_labels={
                        "title": title,
                        "x_axis": x_axis,
                        "y_axis": y_axis,
                        "legend_title": leg_title,
                        "cat_labels": cat_labels,
                        "file_name": file_name
                        })
        return()
    
    def stacked_areachart(self, variables, country, output, negative=[], add_line=None,
                          folder="graphs", figsize= (16,9), show=False, save=True,
                          time_subset=[None, None], typ_day=dict(), typ_week=dict(),
                          label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(variables, list)):
            variables = [variables]
            
        ## Flatten column index over multiindex
        variables = _index_flatselect(variables, self.data)
        
        ## Generate plotting data
        data = self.data[variables+[("index","hour"), ("index","area")]]
        data = data.loc[data[("index","area")] == country,:]
        data.pop(("index","area"))
        
        ## Generate text for title and axes
        if len(variables) == 1:
            var = self.vardesc_df.loc[variables,label_lang].tolist()[0]
            title = "Area chart of "+var+" in "+country
            y_axis = var
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "areachart_"+country+"_"+variables[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
        else:
            title = "Stacked areachart for "+country
            y_axis = "Values"
            if typ_week:
                x_axis = "Hours of week"
            elif typ_day:
                x_axis = "Hours of day"
            else:
                x_axis = "Time"
            file_name = "areachart_"+country+"_".join(list(map(lambda x: x[1], variables)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
        
        ## Call plotting function
        stacked_areachart(data= data, variables=variables, output=output, negative=negative,
                          add_line=add_line, folder=folder, figsize=figsize, show=show,
                          save=save, time_subset=time_subset, typ_day=typ_day,
                          typ_week=typ_week, plot_labels={
                              "title": title,
                              "x_axis": x_axis,
                              "y_axis": y_axis,
                              "legend_title": leg_title,
                              "cat_labels": cat_labels,
                              "file_name": file_name
                              })        
        
        return()
    
    def energy_piechart(self, variables, country, output, folder="graphs", figsize= (16,9),
                        show=False, save=True, time_subset=[None, None], typ_day=dict(),
                        typ_week=dict(), label_lang="desc_en"):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(variables, list)):
            variables = [variables]
            
        ## Flatten column index over multiindex
        variables = _index_flatselect(variables, self.data)
        
        ## Generate plotting data
        data = self.data[variables+[("index","hour"), ("index","area")]]
        data = data.loc[data[("index","area")] == country,:]
        data.pop(("index","area"))
        
        ## Generate text for title and axes
        if len(variables) == 1:
            var = self.vardesc_df.loc[variables,label_lang].tolist()[0]
            title = "Pie chart of "+var+" in "+country
            y_axis = var
            x_axis = "Time"
            file_name = "piechart_"+country+"_"+variables[0][1]
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
        else:
            title = "Pie chart for "+country
            y_axis = "Values"
            x_axis = "Time"
            file_name = "piechart_"+country+"_".join(list(map(lambda x: x[1], variables)))
            leg_title = "Legend"
            cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
 
        ## Call plotting function
        energy_piechart(data= data, variables=variables, output=output, folder=folder,
                        figsize=figsize, show=show, save=save, time_subset=time_subset,
                        typ_day=typ_day, typ_week=typ_week, 
                        plot_labels={
                            "title": title,
                            "x_axis": x_axis,
                            "y_axis": y_axis,
                            "legend_title": leg_title,
                            "cat_labels": cat_labels,
                            "file_name": file_name
                            })

        return()
    
    def reg_energydata(self, dep, indep, country, squared=[], logged=[], constant=True,
                       include_aov=True, time_subset=[None, None],
                       typ_day=dict(), typ_week=dict()):
        
        ## Transform input "variables" into list if not one
        if not(isinstance(indep, list)):
            indep = [indep]
            
        ## Flatten column index over multiindex
        indep = _index_flatselect(indep, self.data)
        
        ## Generate plotting data
        data = self.data[indep+[dep, ("index","hour"), ("index","area")]]
        data = data.loc[data[("index","area")] == country,:]
        data.pop(("index","area"))
        
        ## Call regression function
        reg_energydata(data=data, dep=dep, indep=indep, squared=squared, 
                       logged=logged, constant=constant, include_aov=include_aov,
                       time_subset=time_subset, typ_day=typ_day, typ_week=typ_week)
               
        return()
    
    def energy_balancechart(self, display, country, output, folder="graphs", figsize= (16,9),
                            show=False, save=True, time_subset=[None, None], typ_day=dict(),
                            typ_week=dict(), label_lang="desc_en"):
        
        ## Flatten column index over multiindex
        variables = _index_flatselect(["generation","consumption"], self.data)

        ## Generate plotting data
        data = self.data[variables+[("index","hour"),("index","area")]]
        data = data.loc[data[("index","area")] == country,:]
        data.pop(("index","area"))
        
        ## Generate text for title and axes
        if display == "absolute":
            y_axis = "Energy constitution in GWh"
        elif display == "relative":
            y_axis = "Energy constitution in %"
        title = "Energy balance chart for "+country
        x_axis = ""
        file_name = "energy_balancechart_"+country
        leg_title = "Legend"
        cat_labels = self.vardesc_df.loc[variables,label_lang].tolist()
 
        ## Call plotting function
        stacked_barplot(data= data, x="category", stack_variables="variables", 
                        output=output, display=display , folder=folder, figsize=figsize,
                        show=show, save=save, time_subset=time_subset, typ_day=typ_day,
                        typ_week=typ_week, plot_labels={
                            "title": title,
                            "y_axis": y_axis,
                            "x_axis": x_axis,
                            "legend_title": leg_title,
                            "cat_labels": cat_labels,
                            "file_name": file_name
                            })

        return()
    
import pandas as pd
import eoles_graphics as eg

### Set WD
#os.chdir(r"C:\Users\Nicolas\OneDrive - Zeppelin-University gGmbH\Dokumente\Projekte\CIRED\eoles_visualisation")

#########################################################################################################
######################################## Test base Eoles ################################################
#########################################################################################################

## Define paths for input files
eoles_base_path = r"C:\Users\Nicolas\OneDrive - Zeppelin-University gGmbH\Dokumente\Projekte\CIRED\eoles"

## Initialise model object
test_EL = eg.Eoles_baseline("2006", base_path= eoles_base_path)

## Price visualisations
test_EL.density_plot(variables= [("cost","elec_balance_dual_values")],
                      country="FR", output="png", show=True, save=True)

test_EL.cdf_plot(variables= [("cost","elec_balance_dual_values")],
                  country="FR", output="png", show=True, save=True)

test_EL.price_duration_curve(prices= [("cost","elec_balance_dual_values")],
                              country="FR", output="png", show=True, save=True)

## Produce plots on variables

# Area chart of generation, consumption and prices
test_EL.stacked_areachart(variables= ["generation"], 
                           negative = [('consumption','phs'),
                                       ('consumption','battery')],
                           add_line=("cost","elec_balance_dual_values"),
                           country="FR", output="png", show=True, save=True)

# Energy balance plot generation consumption
test_EL.energy_balancechart(display="absolute", country="FR", output="png", 
                             show=True, save=True)

# Energy piechart of generation sources
test_EL.energy_piechart(variables=["generation"], country="FR", output="png", 
                         show=True, save=True)

# Energy piechart of consumption sources
test_EL.energy_piechart(variables=["consumption"], country="FR", output="png", 
                         show=True, save=True)

## Line plot of typical week of renewables for all year
test_EL.energy_line([("generation","pv"), ("generation","river"),
                      ("generation","lake_phs"), ("generation","wind")], 
                     country="FR",output="png", 
                     typ_week={"quarter":[1,2,3,4]}, show=True, save=True)

## Line plot of typical week of renewables for summer months
test_EL.energy_line([("generation","pv"), ("generation","river"),
                      ("generation","lake_phs"), ("generation","wind")], 
                     country="FR",output="png", 
                     typ_week={"month":[5,6,7,8,9]}, show=True, save=True)

## Line plot of typical day of renewables in december
test_EL.energy_line([("generation","pv"), ("generation","river"),
                      ("generation","lake_phs"), ("generation","wind")], 
                     output="png", 
                     typ_day={"month":[12]}, show=True, save=True)

#########################################################################################################
######################################## Test Multicountry Eoles ########################################
#########################################################################################################

## Define paths for input files
eoles_multicountry_path = r"C:\Users\Nicolas\OneDrive - Zeppelin-University gGmbH\Dokumente\Projekte\CIRED\EOLES-Dispatch"
eoles_historic_path = eoles_multicountry_path + r"\_HISTORIC"

## Initialise model object
test_MLE = eg.Eoles_multicountry(base_path=eoles_multicountry_path,
                                 historic_path=eoles_historic_path)

## Create variable names as english variable descriptions
test_MLE.update_vardesc(changes=pd.DataFrame({"desc_en": test_MLE.data.columns.droplevel(0).to_list()},
                                              index=test_MLE.data.columns))

## Compute additional variables for analysis
test_MLE.compute_analysisvar(variables=["res_demand","diffres_demand","price_error",
                                        "absprice_error","net_IM"])

## Price visualisations - do not call before compute_analysisvar
test_MLE.attach_historic()

test_MLE.density_plot(variables= [("sim_cost","elec_balance_dual_values"),
                                  ("act_cost","elec_balance_dual_values")],
                      country="FR", output="png", show=True, save=True)

test_MLE.cdf_plot(variables= [("sim_cost","elec_balance_dual_values"),
                              ("act_cost","elec_balance_dual_values")],
                  country="FR", output="png", show=True, save=True)

test_MLE.price_duration_curve(prices= [("sim_cost","elec_balance_dual_values"),
                                       ("act_cost","elec_balance_dual_values")],
                              country="FR", output="png", show=True, save=True)

test_MLE.detach_historic()


## Produce plots on variables

# Line plot of price error and residual demand over time
test_MLE.energy_line([("analysis","price_error"), ("analysis","res_demand")], 
                     country="FR",output="png", show=True, save=True)

# Area chart of generation, consumption and prices
test_MLE.stacked_areachart(variables=["generation"], 
                           negative = [('consumption','phs'),
                                       ('consumption','battery'),
                                       ('consumption','net_exports')],
                           add_line=("cost","elec_balance_dual_values"),
                           country="FR", output="png", show=True, save=True)

# Energy balance plot generation consumption
test_MLE.energy_balancechart(display="absolute", country="FR", output="png", 
                             show=True, save=True)

# Energy piechart of generation sources
test_MLE.energy_piechart(variables=["generation"], country="FR", output="png", 
                         show=True, save=True)

# Energy piechart of consumption sources
test_MLE.energy_piechart(variables=["consumption"], country="FR", output="png", 
                         show=True, save=True)

## Line plot of typical week of renewables for all year
test_MLE.energy_line([("generation","pv"), ("generation","river"),
                      ("generation","lake_phs"), ("generation","wind")], 
                     country="FR",output="png", 
                     typ_week={"quarter":[1,2,3,4]}, show=True, save=True)

## Line plot of typical week of renewables for summer months
test_MLE.energy_line([("generation","pv"), ("generation","river"),
                      ("generation","lake_phs"), ("generation","wind")], 
                     country="FR",output="png", 
                     typ_week={"month":[5,6,7,8,9]}, show=True, save=True)

## Line plot of typical day of renewables in december
test_MLE.energy_line([("generation","pv"), ("generation","river"),
                      ("generation","lake_phs"), ("generation","wind")], 
                     country="FR",output="png", 
                     typ_day={"month":[12]}, show=True, save=True)
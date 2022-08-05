import os as os
import eoles_graphics as eg

### Set WD
os.chdir(r"C:\Users\Nicolas\OneDrive - Zeppelin-University gGmbH\Dokumente\Projekte\CIRED\eoles_visualisation")

#########################################################################################################
######################################## Test base Eoles ################################################
#########################################################################################################

test_EL = eg.Eoles_baseline("2006")

test_EL.energy_line([("generation","pv_g"), ("generation","onshore")], output="html",
                    show=False, save=True, typ_week={"quarter":[1,2,3,4]})

test_EL.stacked_areachart([("generation","pv_g"), ("generation","onshore")], 
                          output="html",
                            show=False, save=True, typ_week={"quarter":[1,2,3,4]},
                            add_line = ("consumption","electricity_demand"))

#########################################################################################################
######################################## Test Multicountry Eoles ########################################
#########################################################################################################

test_MLE = eg.Eoles_multicountry(base_path=r"C:\Users\Nicolas\OneDrive - Zeppelin-University gGmbH\Dokumente\Projekte\CIRED\EOLES-Dispatch",
                                 historic_path=r"C:\Users\Nicolas\OneDrive - Zeppelin-University gGmbH\Dokumente\Projekte\CIRED\EOLES-Dispatch\_HISTORIC")

test_MLE.update_vardesc(changes=pd.DataFrame({"desc_en": test_MLE.data.columns.droplevel(0).to_list()},
                                              index=test_MLE.data.columns))

test_MLE.energy_line(("generation","pv"), country="FR",output="png", 
                        typ_week={"quarter":[1,2,3,4]}, save=True)
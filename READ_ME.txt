############################# Eoles Visualisation functions #########################

This packacke provides functionalities to produce data visualisations of the Eoles models easily. The Eoles
model can be used to simulate energy generation price development over a long time, while accounting for 
diverse and stochastically varying energy sources. Functionalities are extended on the baseline Eoles model
as provided by https://gitlab.in2p3.fr/nilam.de_oliveira-gill/eoles and the Multicountry Eoles version from
Clement Leblanc https://github.com/c-leblanc (still under private development). The package has the 
following structure:

- Helper functions for data manipulation & visualisation
- Data loading function for Eoles output and historic data (relies on pd.Multiindex for Baseline)
- Data transformation functions
- Generic visualisation functions for any DataFrame (work on all kinds of index tables)
- Model objects that combine the above for visual production

It is intended that users should primarily rely on the model objects for constructing visuals. Further 
developments specific to a visual from a model obejct or the model object in general can be added there, while
changes to the general functionalities of the visualisation functions should be added in these and hence
become accessible for all model objects.

Pending future developments (collaborations are welcome) are:
- Extend variable description table by color codes
- Allow for color specififcations in generic plot functions
- Add loading function for historic data on baseline Eoles model
- Include richer descriptions for multicountry model through multindex (and hence allow for units)
- Include french descriptions for all variables (Baseline & Multicountry)
- Standardise multicountry model output in terms of units (towards GWh) or include conversion section in load function
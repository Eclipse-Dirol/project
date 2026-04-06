# !!! для FE

- median для LotFrontage, MasVnrArea из nun
- Flag GarageYrBlt -> have_garage, так же нужно учитывать GarageArea, GarageCars | не дропая GarageYrBlt заполнить допустим 'unknown'
- можно выделить наиболее важные в линейной зависимости (числовые):
GarageArea, GarageCars, TotRmsAbvGrd, Fireplaces, FullBath, GrLivArea, 1stFlrSF, TotalBsmtSF, YearBuilt, YearRemodAdd, MasVnrArea
- так же не стоит забывать про 2 этажи, можно попробовать сделать flag: have_second_floor
- можно сделать flag с ремонтом домов и old_house.
- Exterior1st+Exterior2st сделать flag have_Exterior
- сделать flag: have_basement
- сделать flag: have_Fireplaces
- сделать flag: have_WoodDeckSF из **OpenPorchSF** и **WoodDeckSF**
- сделать flag: have_pool | **PoolArea** **PoolQC**
- сделать flag: have_Fence | **Fence**
- можно вынести high_price ~ 0.10 - 0.15 from high
# Сделать unknown для всех nun в category features

1. посчитать всякие groupby по category фичам
2. 
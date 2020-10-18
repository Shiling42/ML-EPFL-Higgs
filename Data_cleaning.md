## Data grouping

The jet_num is an Categorical data, which take integers $0$, $1$, $2$ and $3$. This integers represent the number of jet particles. Physicall speaking, this number indicate the fact that .

Theremofer, we can use this catagorical data to group other feature. In the dataset, there are many nulls, shown as $-999$. By inspecting, we find these nulls is related to the jet number. The jet numebr is 

How to deal with missing feature: imputation. Mean imputation is the simplest way.

DER_mass_MMC is important, which is the expected mass, and we can filter them by regrouping.
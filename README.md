# pythoninterestrates (Under Construction)
Interest rate modeling tools
Includes tools for fitting interest models, simulating interest models and discounting instruments.
The project also includes various interest rate instruments; look in interest_rate_instruments.py source. A supported can also be found interest_rate_base.py (rate_instruments enumeration). 
   This additionally incorporates a first attempt to decompose cash flows into most basic pieces
incoporation into larger portfolios of instruments. 

## Instrument Pricing Logic 
The base logic is that instruments include base pricing (simple) methodology, e.g. Black or Bachelier, which can be overriden with model parameter, e.g hjm_model. An example HJM implementation can be found for VaSicek model. 

## Curve Constructor 
Incoporates methods "EXACT" fitting methods for "Boot-Strapping" and Pseudo Inverse. Example specifications 
can be foudn in the test/data json files for which tests exist in test directory.
   Successful Curver Constructor object construction contains TWO items  1.) results (pd.DataFram) dataframe of 
instrument zeros and 2.) zeros (object of class discount_calculator) with matrix of zeros for time period in cf_matrix
cash flows. The results dataframe contains maturities, zeros, forwards and yields for json specified instruments (in
case of Pseudo-Inverse) and additionally bootstrapped instruments (in case of Boot-Strap method).

## Lorimier method (current dev!)
* Inherits from  Curve Constructor method. Applies spline to interpolate between yield nodes. Will then calculate
  - forwards from yields assuming continous compounding.
* Problems
 - Zero Coupon bond calculation, interpetation and application
 - automatic choice of estimation set
 - include zero coppon bonds new set and run first step of init
 - matrix construction (can I replace in the __init__ call of parent)
 - estimation process
 - fit review & analysis

### Releases 
* __0.5__:  Added coupon bonds as element from which can derive Pseudo-Inverse fitting. Added control swaps
  - starting t0=T0 swap and curve_constructor implemementations.   

## Testing
I have started writing testing, but this peice is under current construction. The testing is based on pytest and can be 
accessed from test directory with call to pytest. 

## Examples
Look into test/data for example assumptions json that cna be used to fit interest models. Take note of the method indicator. Method = 0 => simple bootstrap, while Method=1 => Pseudo inverse. This impacts the json specification. 


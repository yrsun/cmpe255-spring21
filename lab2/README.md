# Output:

       engine_cylinders transmission_type      driven_wheels  number_of_doors  \
2779                4.0         automatic   four_wheel_drive              4.0   
3708                4.0         automatic    all_wheel_drive              4.0   
4794                4.0            manual  front_wheel_drive              4.0   
10498               6.0         automatic  front_wheel_drive              4.0   
1880                4.0         automatic  front_wheel_drive              2.0   

      market_category vehicle_size        vehicle_style  highway_mpg  \
2779              NaN      compact  extended_cab_pickup           25   
3708           luxury      midsize                sedan           29   
4794        flex_fuel      compact                sedan           36   
10498          luxury      midsize                sedan           34   
1880              NaN      compact          convertible           34   

       city_mpg  popularity          msrp  msrp_pred  
2779         19        1385  5.713415e+23      26885  
3708         22         617  7.148573e+12      54650  
4794         26        5657  1.831423e+04      16775  
10498        21         204  1.266845e+06      42600  
1880         25         873  3.168485e+13      25995  

# Lab 2 - Linear Regression


* 20% of data goes to validation,
* 20% goes to test, and
* the remaining 60% goes to train.

```
y = g(x) = g(x1, x2, ..., xn) = w0 + x1 w1 + x2 w2 + ... + xn wn
```

Bias term is the value we would predict if we did not know anything about the car; it serves as a baseline.


## Final Output

Print out 5 cars' original msrp vs. predicted msrp

| engine_cylinders	| transmission_type	| driven_wheels	| number_of_doors	| market_category	| vehicle_size |	vehicle_style |	highway_mpg	| city_mpg |	popularity | msrp | msrp_pred |
|---------|-------|-------|---------|--------|-------|-------|------|-----|------|-----|---|
|x | x| x | x | x | x | x |  x | x | x | x |

rdata = prnist([0:9],[1:100:1000]);
a = my_rep(rdata);
w = pcam([],0.85) * ldc;
w = a*w;
e = nist_eval('my_rep', w)
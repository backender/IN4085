rdata = prnist([0:9],[1:2:1000]);
a = my_rep(rdata);
w = pcam([],0.85) * svc(proxm('p',5));
w = a*w;
e = nist_eval('my_rep', w)
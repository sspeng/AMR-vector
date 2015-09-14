#!/usr/bin/expect

set login "salaza11"
set addr "golub.campuscluster.illinois.edu"
set pw "balbarrN1"

spawn scp general.in femparameters.h femparameters.C TopOpt.h TopOpt.C vonmises.h vonmises.C compliance_traction.C compliance_traction.h compliance.C compliance.h vector_fe_ex1.C resder.h resder.C $login@$addr:/home/$login/AdaptiveTopOpt
expect "$login@$addr\'s password:"
send "$pw\r"
expect "#"
interact

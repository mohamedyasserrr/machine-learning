clear all
ds = datastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
size(T);
Alpha=.01;
m=length(T{:,1});
U0=T{:,14};
U=T{:,1:13};
U2=U.^2;
U3=sqrt(U);
 
X=[ones(m,1) U U.^2]; %third hypothesis

n=length(F(1,:));

for w=2:n
    if max(abs(F(:,w)))~=0
    F(:,w)=(F(:,w)-mean((F(:,w))))./std(F(:,w));
    end
end

P=T{:,14};
TH=zeros(n,1);
k=1;
G=(1./(1+exp(-F*TH)));
E(k)=(1/(m))*sum((-P.'*log(1./(1+exp(-F*TH))))-((1-P).'*log(1-(1./(1+exp(-F*TH))))));

R=1;
while R==1
Alpha=Alpha*1;
TH=TH-(Alpha/m)*F'*((1./(1+exp(-F*TH)))-P);
k=k+1;
E(k)=(1/(m))*sum((-P.'*log(1./(1+exp(-F*TH))))-((1-P).'*log(1-(1./(1+exp(-F*TH))))));
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.0001;
    R=0;
end
end
plot(1:k,E)
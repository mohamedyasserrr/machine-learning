clear all;
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
size(T);
Alpha=.01;

m=length(T{:,1});
U0=T{:,2};
U=T{:,4:19};

U1=T{:,20:21};
U2=U.^2;
U3=sqrt(U);

X=[ones(m,1) U U1 U.^2 U.^3];%first hypothesis

n=length(F(1,:));

for w=2:n
    if max(abs(F(:,w)))~=0
    F(:,w)=(F(:,w)-mean((F(:,w))))./std(F(:,w));
    end
end


P=T{:,3}/mean(T{:,3});
Th=zeros(n,1);
k=1;

E(k)=(1/(2*m))*sum((F*Th-P).^2);


R=1;
while R==1
Alpha=Alpha*1;
Th=Th-(Alpha/m)*F'*(F*Th-P);
k=k+1;
E(k)=(1/(2*m))*sum((F*Th-P).^2);
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.0001;
    R=0;
end
end
plot(1:k,E)
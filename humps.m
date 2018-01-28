I = eye(3);
% W = 1.5*eye(3) + 0.4*sign(randn(3)).*(ones(3) - I);
W = 1.5*eye(3) + 0.2*ones(3).*(ones(3) - I);
disp(W)

a = -1:0.01:1;
a = a(2:end-1); % avoid inf at +/- 1
z = atanh(ones(3,1)*a) - diag(W)*a; % zeta

i = 1;
w_ = (W(i,:) - W(i,i)*I(i,:))';
s = z(i,:) ./ (w_'*w_);
a_ = I(i,:)'*a;
sw_ = w_*s;

n = null([I(i,:); w_']);
n_ = n*ones(1,numel(a));
V = [a_ + sw_ + n_, a_ + sw_ - n_];
F = [1:numel(a)-1; numel(a) + (1:numel(a)-1); numel(a) + (2:numel(a)); 2:numel(a)];

a_t = sqrt(1 - 1/diag(W));
z_t = atanh(a_t) - diag(W).*a_t;
s_t(i) = z_t(i) ./ (w_'*w_);

cla; hold on;
patch('Faces',F', 'Vertices', V', 'FaceColor',[.5 .5 .5], 'EdgeColor', 'none');
quiver3(0,0,0,I(i,1),I(i,2),I(i,3),0,'k');
quiver3(0,0,0,-I(i,1),-I(i,2),-I(i,3),0,'k');
quiver3(-a_t(i)*I(i,1),-a_t(i)*I(i,2),-a_t(i)*I(i,3),-s_t(i)*w_(1),-s_t(i)*w_(2),-s_t(i)*w_(3),0,'k');
% quiver3(0,0,0,W(i,1),W(i,2),W(i,3),0,'k');
text(-a_t(i)*I(i,1),-a_t(i)*I(i,2),-a_t(i)*I(i,3),{'a'},'FontSize',20);
light

xlim([-1,1])
ylim([-1,1])
zlim([-1,1])
view([-185, 30])
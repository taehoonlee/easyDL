function ttttt = easyDLim2col(a, block, f)

    R = a(1);
    C = a(2);
    F = a(3);
    M = a(4);
    
    r = block(1);
    c = block(2);
    nr = R-r+1; nc = C-c+1;
    cidx = (0:r-1)'; ridx = 1:nr;
    t = cidx(:,ones(nr,1)) + ridx(ones(r,1),:);
    tt = zeros(r*c,nr);
    rows = 1:r;
    for i=0:c-1,
        tt(i*r+rows,:) = t+R*i;
    end
    if f
        tt = flip(tt, 1);
    end
    ttt = zeros(r*c,nr*nc);
    cols = 1:size(tt,2);
    for j=0:nc-1,
        ttt(:,j*nr+cols) = tt+R*j;
    end
    tttt = zeros(r*c*F,nr*nc);
    nx = r*c;
    ndim = R*C;
    xs = 1:size(ttt,1);
    for j=0:F-1,
        tttt(j*nx+xs,:) = ttt+ndim*j;
    end
    ttttt = zeros(r*c*F,nr*nc*M);
    ny = nr*nc;
    ndim = R*C*F;
    ys = 1:size(ttt,2);
    for j=0:M-1,
        ttttt(:,j*ny+ys) = tttt+ndim*j;
    end
    %b = a(ttttt);
/* Live empowerment-gain agent — the same drive as the paper, running in-browser.
   No playback: the planner, body bottleneck, and forgetting model all run here. */
(function () {
  const D = window.EMP_DATA;
  if (!D) { console.error("EMP_DATA missing"); return; }
  const S = D.S, RES = D.resource;

  // ---- colormap (matches the paper's EMAP) ----
  const STOPS = [[16,19,43],[38,65,143],[31,158,137],[127,211,78],[246,226,122]];
  function emap(t){ t=Math.max(0,Math.min(1,t)); const x=t*(STOPS.length-1);
    const i=Math.floor(x), f=x-i, a=STOPS[i], b=STOPS[Math.min(i+1,STOPS.length-1)];
    return `rgb(${a[0]+(b[0]-a[0])*f|0},${a[1]+(b[1]-a[1])*f|0},${a[2]+(b[2]-a[2])*f|0})`; }

  const clip = v => v<0?0:(v>1?1:v);
  const gNoise = b => 1/(1+Math.exp(D.g_k*(b-D.g_b0)));
  const NB = D.b_bins.length - 1;
  function Eval(cell,b){ return D.Etable[Math.round(clip(b)*NB)][cell]; }

  // ---- state ----
  const P = { delta:0.006, intake:0.04, lam:0.996, gamma:0.94 };
  let s, b, N, lastFeed, intervals, faceHist, bHist, running, speed;
  function reset(){
    s=RES; b=1.0; N=new Float64Array(S);
    lastFeed=0; intervals=[]; faceHist=[]; bHist=[]; step=0; ventering=false;
    sumSom=0; sumEpi=0;
  }
  let step=0, sumSom=0, sumEpi=0, ventering=false;

  // ---- one planning + dynamics step ----
  function tick(){
    const dist = D.dist[s];
    let bestVal=-Infinity, bestC=s, somSel=0, epiSel=0;
    for(let c=0;c<S;c++){
      const tau=dist[c]; if(tau>=999) continue;
      const disc=Math.pow(P.gamma,tau);
      let som=0;
      if(c===RES){ som=Math.max(0, D.Emax[RES]-Eval(RES, clip(b-P.delta*tau))); }
      const dHome=D.dist[c][RES];
      const bviab=clip(b-P.delta*(tau+dHome));
      const u=1/(1+N[c]);
      const epi=u*Eval(c,bviab);
      const val=disc*(som+epi);
      if(val>bestVal){ bestVal=val; bestC=c; somSel=disc*som; epiSel=disc*epi; }
    }
    let a = (bestC===s)?4:D.first_step[s][bestC]; if(a<0) a=4;
    if(Math.random()<gNoise(b)) a=Math.floor(Math.random()*5);
    let ns = D.step[s][a];

    b -= P.delta;
    if(ns===RES){ b=Math.min(1,b+P.intake+P.delta); }
    b=clip(b);
    for(let c=0;c<S;c++) N[c]*=P.lam;
    N[ns]+=1;

    // feeding-interval bookkeeping (return to resource after >=6 steps away)
    if(ns===RES && step-lastFeed>=6){ intervals.push(step-lastFeed);
      if(intervals.length>12) intervals.shift(); lastFeed=step; }
    else if(ns===RES){ lastFeed=step; }

    s=ns; step++;
    const phase = somSel>=epiSel ? "consolidate" : "venture";
    ventering = phase==="venture";
    sumSom=somSel; sumEpi=epiSel;
    faceHist.push([somSel,epiSel]); if(faceHist.length>140) faceHist.shift();
    bHist.push([b,ventering]); if(bHist.length>260) bHist.shift();
    return {bestC, somSel, epiSel, phase};
  }

  // ---- rendering ----
  const arena=document.getElementById("arena"), ax=arena.getContext("2d");
  const pp=document.getElementById("phaseplane"), px=pp.getContext("2d");
  const strip=document.getElementById("bstrip"), sx=strip.getContext("2d");
  const CELL=Math.floor(Math.min(arena.width/D.cols, arena.height/D.rows));
  const OX=(arena.width-CELL*D.cols)/2, OY=(arena.height-CELL*D.rows)/2;
  const trail=[];

  function drawArena(){
    ax.clearRect(0,0,arena.width,arena.height);
    for(let c=0;c<S;c++){
      const [r,cc]=D.cells[c];
      const x=OX+cc*CELL, y=OY+r*CELL;
      ax.fillStyle=emap(Eval(c,b)/4.6);
      ax.fillRect(x,y,CELL-1,CELL-1);
      const u=1/(1+N[c]);                       // novelty overlay
      if(u>0.18){ ax.fillStyle=`rgba(255,255,255,${0.5*u})`;
        ax.beginPath(); ax.arc(x+CELL/2,y+CELL/2,1.6,0,7); ax.fill(); }
    }
    // resource star
    const [rr,rc]=D.resource_rc, sxp=OX+rc*CELL+CELL/2, syp=OY+rr*CELL+CELL/2;
    ax.fillStyle="#fff"; ax.strokeStyle="#1a1a2e"; ax.lineWidth=1; star(ax,sxp,syp,CELL*0.34);
    // trail
    trail.push([s]); if(trail.length>26) trail.shift();
    trail.forEach((t,i)=>{ const [r,cc]=D.cells[t[0]];
      ax.fillStyle=`rgba(233,69,96,${0.05+0.5*i/trail.length})`;
      ax.beginPath(); ax.arc(OX+cc*CELL+CELL/2,OY+r*CELL+CELL/2,CELL*0.18,0,7); ax.fill(); });
    // agent
    const [ar,ac]=D.cells[s];
    ax.fillStyle= ventering ? "#e94560" : "#0f7173";
    ax.strokeStyle="#fff"; ax.lineWidth=2;
    ax.beginPath(); ax.arc(OX+ac*CELL+CELL/2,OY+ar*CELL+CELL/2,CELL*0.32,0,7); ax.fill(); ax.stroke();
  }
  function star(ctx,cx,cy,R){ ctx.beginPath();
    for(let i=0;i<10;i++){ const a=Math.PI/5*i-Math.PI/2, r=i%2?R*0.45:R;
      ctx[i?"lineTo":"moveTo"](cx+r*Math.cos(a),cy+r*Math.sin(a)); }
    ctx.closePath(); ctx.fill(); ctx.stroke(); }

  function drawPlane(){
    px.clearRect(0,0,pp.width,pp.height);
    const W=pp.width,H=pp.height, mx=3.8, my=2.6;
    px.strokeStyle="#ddd"; px.lineWidth=1;
    px.beginPath(); px.moveTo(34,H-26); px.lineTo(W-8,H-26); px.lineTo(W-8,8); px.stroke();
    px.fillStyle="#888"; px.font="10px sans-serif";
    px.fillText("somatic →",W-66,H-12); px.save(); px.translate(12,H-30); px.rotate(-Math.PI/2);
    px.fillText("epistemic →",0,0); px.restore();
    faceHist.forEach((p,i)=>{ const x=34+(p[0]/mx)*(W-44), y=(H-26)-(p[1]/my)*(H-36);
      px.fillStyle=`rgba(${p[1]>p[0]?"233,69,96":"15,113,115"},${0.15+0.7*i/faceHist.length})`;
      px.beginPath(); px.arc(x,y,2.6,0,7); px.fill(); });
  }
  function drawStrip(){
    sx.clearRect(0,0,strip.width,strip.height);
    const W=strip.width,H=strip.height;
    bHist.forEach((p,i)=>{ const x=i/260*W;
      if(p[1]){ sx.fillStyle="rgba(233,69,96,0.10)"; sx.fillRect(x,0,W/260+1,H); } });
    sx.strokeStyle="#1a1a2e"; sx.lineWidth=1.4; sx.beginPath();
    bHist.forEach((p,i)=>{ const x=i/260*W, y=H-(p[0]*0.85+0.07)*H;
      sx[i?"lineTo":"moveTo"](x,y); }); sx.stroke();
    sx.fillStyle="#888"; sx.font="10px sans-serif"; sx.fillText("body integrity b(t)",6,12);
  }

  // ---- HUD ----
  const el=id=>document.getElementById(id);
  function hud(){
    el("phaseBadge").textContent = ventering ? "VENTURING" : "CONSOLIDATING";
    el("phaseBadge").className = "badge " + (ventering?"vent":"cons");
    el("bbar").style.width = (b*100).toFixed(0)+"%";
    el("bbar").style.background = b<0.4 ? "#e94560" : (b<0.7?"#d98e04":"#0f7173");
    el("bval").textContent = b.toFixed(2);
    const tot=sumSom+sumEpi+1e-9;
    el("sombar").style.width = (100*sumSom/Math.max(tot,sumSom,sumEpi)).toFixed(0)+"%";
    el("epibar").style.width = (100*sumEpi/Math.max(tot,sumSom,sumEpi)).toFixed(0)+"%";
    const Tpred = 0.14/P.delta;
    const Tmeas = intervals.length>1 ? intervals.reduce((a,c)=>a+c,0)/intervals.length : NaN;
    el("tpred").textContent = Tpred.toFixed(0);
    el("tmeas").textContent = isNaN(Tmeas) ? "—" : Tmeas.toFixed(0);
  }

  // ---- loop ----
  let acc=0;
  function frame(){
    if(running){ const n=speed; for(let k=0;k<n;k++) tick(); }
    drawArena(); drawPlane(); drawStrip(); hud();
    requestAnimationFrame(frame);
  }

  // ---- controls ----
  reset(); running=true; speed=2;
  function bind(id,key,fmt){ const sl=el(id), out=el(id+"v");
    const upd=()=>{ P[key]=parseFloat(sl.value); out.textContent=fmt(P[key]); };
    sl.addEventListener("input",upd); upd(); }
  bind("delta","delta",v=>v.toFixed(3));
  bind("lam","lam",v=>v.toFixed(3));
  el("speed").addEventListener("input",e=>{ speed=parseInt(e.target.value);
    el("speedv").textContent=speed+"×"; });
  el("speedv").textContent="2×";
  el("playBtn").addEventListener("click",e=>{ running=!running;
    e.target.textContent=running?"⏸ pause":"▶ play"; });
  el("resetBtn").addEventListener("click",()=>{ reset(); trail.length=0; });
  // debug hook: drive the model without rendering (works even in a hidden tab)
  window.empDebug = { run(n){ let v=0,c=0; for(let i=0;i<n;i++){ const r=tick();
      if(r.phase==="venture") v++; else c++; }
    return {step, b:+b.toFixed(3), cell:s, vent:v, cons:c,
            intervals:intervals.slice(), Tpred:+(0.14/P.delta).toFixed(1)}; },
    setDelta(d){ P.delta=d; } };
  frame();
})();

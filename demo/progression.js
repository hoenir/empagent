/* Live demo: empowerment learning-progress vs surprise, in the equidistant arena.
   A faithful lightweight model -- structural empowerment ceiling C[c] (precomputed),
   epistemic uncertainty that falls with visits, and an aleatoric noise floor in the
   noise chamber that does NOT. Watch the surprise agent get trapped on the noisy TV. */
(function () {
  const D = window.PROG_DATA;
  if (!D) { console.error("PROG_DATA missing"); return; }
  const S = D.S, RES = D.resource;
  const STOPS = [[16,19,43],[38,65,143],[31,158,137],[127,211,78],[246,226,122]];
  function emap(t){ t=Math.max(0,Math.min(1,t)); const x=t*(STOPS.length-1);
    const i=Math.floor(x), f=x-i, a=STOPS[i], b=STOPS[Math.min(i+1,STOPS.length-1)];
    return `rgb(${a[0]+(b[0]-a[0])*f|0},${a[1]+(b[1]-a[1])*f|0},${a[2]+(b[2]-a[2])*f|0})`; }

  const P = { mode:"lp", noise:0.5, gamma:0.97, eps:0.3, LR:0.5, alpha:0.35 };
  let N, Ehat, prevE, LP, s, step, dwell;
  function reset(){
    N=new Float64Array(S); Ehat=new Float64Array(S); prevE=new Float64Array(S);
    LP=new Float64Array(S).fill(0.6); s=RES; step=0; dwell={}; trail.length=0;
  }
  const effNoise = c => (D.region[c]==="noise" ? P.noise : 0);

  function signals(){
    for(let c=0;c<S;c++){
      const ue = 1/(1+N[c]*P.LR);
      Ehat[c] = D.Emax[c]*(1-effNoise(c))*(1-ue);
    }
  }
  function uVal(c){ return Math.max(1/(1+N[c]*P.LR), effNoise(c)); }   // surprise

  function tick(){
    signals();
    // drive per cell
    const dist=D.dist[s];
    let best=-1, bestC=s;
    for(let c=0;c<S;c++){
      const tau=dist[c]; if(tau>=999) continue;
      let drv;
      if(P.mode==="lp"){ drv = (N[c]<1 ? 0.6 : LP[c]); }     // optimism for unvisited
      else { drv = uVal(c); }
      const v = Math.pow(P.gamma,tau)*drv;
      if(v>best){ best=v; bestC=c; }
    }
    let a = (bestC===s)?4:D.first_step[s][bestC]; if(a<0) a=4;
    if(Math.random()<P.eps) a=Math.floor(Math.random()*5);    // babble
    let ns = D.step[s][a];
    if(Math.random()<effNoise(s)) ns = D.step[s][Math.floor(Math.random()*5)]; // env noise
    // learn
    N[ns]+=1;
    signals();
    LP[ns] = (1-P.alpha)*LP[ns] + P.alpha*Math.max(0, Ehat[ns]-prevE[ns]);
    prevE[ns]=Ehat[ns];
    const lab=D.region[ns]||"corridor"; dwell[lab]=(dwell[lab]||0)+1;
    s=ns; step++;
  }

  // ---- render ----
  const cv=document.getElementById("progArena"), cx=cv.getContext("2d");
  const CELL=Math.floor(Math.min(cv.width/D.cols, cv.height/D.rows));
  const OX=(cv.width-CELL*D.cols)/2, OY=(cv.height-CELL*D.rows)/2;
  const trail=[];
  const free=new Set(D.cells.map((rc,i)=>rc[0]*D.cols+rc[1]));
  function draw(){
    cx.clearRect(0,0,cv.width,cv.height);
    for(let c=0;c<S;c++){
      const [r,cc]=D.cells[c], x=OX+cc*CELL, y=OY+r*CELL;
      const sig = P.mode==="lp" ? Math.min(1,(N[c]<1?0.6:LP[c])*3) : uVal(c);
      cx.fillStyle=emap(sig); cx.fillRect(x,y,CELL-1,CELL-1);
    }
    // chamber labels
    cx.fillStyle="#fff"; cx.font="9px sans-serif"; cx.textAlign="center";
    cx.fillText("frontier",OX+7*CELL+CELL/2,OY+2*CELL);
    cx.fillText("frontier",OX+7*CELL+CELL/2,OY+12.6*CELL);
    cx.fillStyle="#ffd2da"; cx.fillText("NOISY TV",OX+12*CELL+CELL/2,OY+7*CELL-6);
    cx.fillStyle="#fff"; cx.fillText("easy",OX+2*CELL+CELL/2,OY+6.4*CELL);
    // resource
    const [rr,rc]=D.cells[RES]; cx.fillStyle="#fff"; cx.strokeStyle="#1a1a2e";
    cx.beginPath(); cx.arc(OX+rc*CELL+CELL/2,OY+rr*CELL+CELL/2,3,0,7); cx.fill();
    // agent
    trail.push(s); if(trail.length>18) trail.shift();
    trail.forEach((t,i)=>{ const [r,cc]=D.cells[t];
      cx.fillStyle=`rgba(255,255,255,${0.05+0.4*i/trail.length})`;
      cx.beginPath(); cx.arc(OX+cc*CELL+CELL/2,OY+r*CELL+CELL/2,CELL*0.16,0,7); cx.fill(); });
    const [ar,ac]=D.cells[s];
    cx.fillStyle = P.mode==="lp" ? "#0f7173" : "#e94560";
    cx.strokeStyle="#fff"; cx.lineWidth=2;
    cx.beginPath(); cx.arc(OX+ac*CELL+CELL/2,OY+ar*CELL+CELL/2,CELL*0.34,0,7); cx.fill(); cx.stroke();
  }

  const el=id=>document.getElementById(id);
  function hud(){
    const tot=Object.values(dwell).reduce((a,c)=>a+c,0)||1;
    const noiseP=100*(dwell["noise"]||0)/tot;
    const learnP=100*((dwell["frontier"]||0)+(dwell["frontier2"]||0)+(dwell["easy"]||0))/tot;
    el("noiseBar").style.width=noiseP.toFixed(0)+"%";
    el("learnBar").style.width=learnP.toFixed(0)+"%";
    el("noiseP").textContent=noiseP.toFixed(0)+"%";
    el("learnP").textContent=learnP.toFixed(0)+"%";
    el("verdict").textContent = P.mode==="lp"
      ? "learning-progress: forages what it can control"
      : (noiseP>40 ? "surprise: trapped on the noisy TV" : "surprise: heading for the noise…");
    el("verdict").style.color = P.mode==="lp" ? "#5fd0c8" : "#ff8a9e";
  }
  let running=true, speed=2;
  function frame(){ if(running){ for(let k=0;k<speed;k++) tick(); } draw(); hud(); requestAnimationFrame(frame); }

  reset();
  el("progMode").addEventListener("change",e=>{ P.mode=e.target.value; reset(); });
  el("progNoise").addEventListener("input",e=>{ P.noise=parseFloat(e.target.value);
    el("progNoisev").textContent=P.noise.toFixed(2); });
  el("progPlay").addEventListener("click",e=>{ running=!running; e.target.textContent=running?"⏸ pause":"▶ play"; });
  el("progReset").addEventListener("click",reset);
  window.progDebug={ run(n){ for(let i=0;i<n;i++) tick(); const t=Object.values(dwell).reduce((a,c)=>a+c,0)||1;
    return {step, noisePct:Math.round(100*(dwell["noise"]||0)/t),
      learnPct:Math.round(100*((dwell["frontier"]||0)+(dwell["frontier2"]||0)+(dwell["easy"]||0))/t)}; },
    setMode(m){P.mode=m;reset();} };
  frame();
})();

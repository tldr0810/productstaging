export interface LocalStageResult {
  staged: string;
  cutout: string;
  mask: string;
  comparison: string;
  cutoutBackend: string;
  sceneBackend: string;
}

const loadImage = (file: File): Promise<HTMLImageElement> => new Promise((resolve, reject) => {
  const image = new Image();
  image.onload = () => resolve(image);
  image.onerror = () => reject(new Error('Could not decode the image.'));
  image.src = URL.createObjectURL(file);
});

const canvas = (width: number, height: number): HTMLCanvasElement => {
  const output = document.createElement('canvas');
  output.width = width;
  output.height = height;
  return output;
};

const png = (input: HTMLCanvasElement): string => input.toDataURL('image/png').split(',')[1];

function backgroundKind(prompt: string): 'wood' | 'marble' | 'outdoor' | 'studio' {
  const value = prompt.toLowerCase();
  if (/(marble|counter|kitchen)/.test(value)) return 'marble';
  if (/(outdoor|garden|patio|grass|sun)/.test(value)) return 'outdoor';
  if (/(studio|seamless|catalog)/.test(value)) return 'studio';
  return 'wood';
}

function drawScene(ctx: CanvasRenderingContext2D, width: number, height: number, prompt: string) {
  const kind = backgroundKind(prompt);
  const top = kind === 'marble' ? '#d7dad7' : kind === 'outdoor' ? '#a9c9d7' : kind === 'studio' ? '#f2f2ee' : '#c49a6c';
  const bottom = kind === 'marble' ? '#9da3a1' : kind === 'outdoor' ? '#5b8047' : kind === 'studio' ? '#d0d1ce' : '#6e482c';
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, top);
  gradient.addColorStop(kind === 'outdoor' ? 0.55 : 0.48, top);
  gradient.addColorStop(1, bottom);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  const horizon = Math.round(height * (kind === 'outdoor' ? 0.55 : 0.48));
  ctx.globalAlpha = 0.24;
  if (kind === 'wood') {
    ctx.strokeStyle = '#3f2819';
    for (let y = horizon + 8; y < height; y += 11) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y + ((y * 13) % 7) - 3); ctx.stroke();
    }
  } else if (kind === 'marble') {
    ctx.strokeStyle = '#ffffff';
    for (let x = -width; x < width * 2; x += 90) {
      ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x, height); ctx.lineTo(x + 210, horizon); ctx.stroke();
    }
  } else if (kind === 'outdoor') {
    ctx.fillStyle = '#315b2b'; ctx.fillRect(0, horizon, width, height - horizon);
    ctx.strokeStyle = '#f6f2d8'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(0, horizon); ctx.lineTo(width, horizon); ctx.stroke();
  } else {
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(0, horizon); ctx.lineTo(width, horizon); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  const light = ctx.createRadialGradient(width * 0.22, height * 0.08, 4, width * 0.22, height * 0.08, width * 0.8);
  light.addColorStop(0, 'rgba(255,245,218,.34)');
  light.addColorStop(1, 'rgba(255,245,218,0)');
  ctx.fillStyle = light;
  ctx.fillRect(0, 0, width, height);
}

export async function stageInBrowser(file: File, prompt: string): Promise<LocalStageResult> {
  const image = await loadImage(file);
  URL.revokeObjectURL(image.src);
  const scale = Math.min(1, 1024 / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(512, Math.round(image.naturalWidth * scale));
  const height = Math.max(512, Math.round(image.naturalHeight * scale));
  const source = canvas(width, height);
  const sourceContext = source.getContext('2d')!;
  sourceContext.drawImage(image, 0, 0, width, height);
  const pixels = sourceContext.getImageData(0, 0, width, height);
  const sample: number[] = [];
  for (let x = 0; x < width; x += Math.max(1, Math.floor(width / 32))) {
    sample.push(x * 4, (height - 1) * width * 4 + x * 4);
  }
  for (let y = 0; y < height; y += Math.max(1, Math.floor(height / 32))) {
    sample.push(y * width * 4, y * width * 4 + (width - 1) * 4);
  }
  const background = [0, 1, 2].map((channel) => sample.reduce((sum, index) => sum + pixels.data[index + channel], 0) / sample.length);
  const alpha = new Uint8ClampedArray(width * height);
  let left = width, top = height, right = 0, bottom = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 4;
      const distance = Math.hypot(pixels.data[index] - background[0], pixels.data[index + 1] - background[1], pixels.data[index + 2] - background[2]);
      const brightness = (pixels.data[index] + pixels.data[index + 1] + pixels.data[index + 2]) / 3;
      const saturation = Math.max(pixels.data[index], pixels.data[index + 1], pixels.data[index + 2]) - Math.min(pixels.data[index], pixels.data[index + 1], pixels.data[index + 2]);
      const foreground = distance > 30 || brightness < 235 || saturation > 30;
      alpha[y * width + x] = foreground ? 255 : 0;
      if (foreground) { left = Math.min(left, x); top = Math.min(top, y); right = Math.max(right, x); bottom = Math.max(bottom, y); }
    }
  }
  if (right <= left || bottom <= top) throw new Error('No foreground detected. Try a product photo with a simple background.');
  const cutout = canvas(width, height);
  const cutoutContext = cutout.getContext('2d')!;
  const cutoutPixels = cutoutContext.createImageData(pixels);
  cutoutPixels.data.set(pixels.data);
  for (let i = 0; i < alpha.length; i++) cutoutPixels.data[i * 4 + 3] = alpha[i];
  cutoutContext.putImageData(cutoutPixels, 0, 0);
  const mask = canvas(width, height);
  const maskContext = mask.getContext('2d')!;
  const maskPixels = maskContext.createImageData(width, height);
  for (let i = 0; i < alpha.length; i++) maskPixels.data.set([alpha[i], alpha[i], alpha[i], 255], i * 4);
  maskContext.putImageData(maskPixels, 0, 0);

  const staged = canvas(width, height);
  const stagedContext = staged.getContext('2d')!;
  drawScene(stagedContext, width, height, prompt);
  const productHeight = Math.round(height * 0.52);
  const productWidth = Math.round((right - left + 1) * productHeight / (bottom - top + 1));
  const x = Math.round((width - productWidth) / 2);
  const y = Math.max(0, Math.round(height * 0.55 - productHeight / 2));
  stagedContext.save();
  stagedContext.globalAlpha = 0.28;
  stagedContext.filter = 'blur(18px)';
  stagedContext.fillStyle = '#20150e';
  stagedContext.beginPath();
  stagedContext.ellipse(x + productWidth / 2, y + productHeight + 13, productWidth * 0.43, productWidth * 0.08, 0, 0, Math.PI * 2);
  stagedContext.fill();
  stagedContext.restore();
  stagedContext.drawImage(cutout, left, top, right - left + 1, bottom - top + 1, x, y, productWidth, productHeight);
  const comparison = canvas(width * 2, height + 54);
  const comparisonContext = comparison.getContext('2d')!;
  comparisonContext.fillStyle = '#f4f2ee'; comparisonContext.fillRect(0, 0, width * 2, height + 54);
  comparisonContext.fillStyle = '#24262a'; comparisonContext.font = '12px system-ui'; comparisonContext.fillText('ORIGINAL', 18, 24); comparisonContext.fillText('STAGED', width + 18, 24);
  comparisonContext.drawImage(source, 0, 54); comparisonContext.drawImage(staged, width, 54);
  return { staged: png(staged), cutout: png(cutout), mask: png(mask), comparison: png(comparison), cutoutBackend: 'browser-simple-mask', sceneBackend: 'browser-scene-fallback' };
}

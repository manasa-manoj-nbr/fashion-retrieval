const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  LevelFormat, ExternalHyperlink,
} = require("docx");

// ---------- small helpers ---------------------------------------------------
const GAP = 120;

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 140 },
    children: [new TextRun({ text })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 100 },
    children: [new TextRun({ text })] });
}
function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: GAP, line: 276 }, alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, ...opts })] });
}
// paragraph from mixed runs
function pr(runs) {
  return new Paragraph({ spacing: { after: GAP, line: 276 }, alignment: AlignmentType.JUSTIFIED, children: runs });
}
function bullet(text, bold) {
  const children = bold
    ? [new TextRun({ text: bold, bold: true }), new TextRun({ text: text })]
    : [new TextRun({ text })];
  return new Paragraph({ numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60, line: 276 }, children });
}
function code(text) {
  return new Paragraph({ spacing: { after: GAP }, shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
    children: [new TextRun({ text, font: "Consolas", size: 18 })] });
}
// "paste your screenshot here" callout box
function placeholder(title, body) {
  const cell = new TableCell({
    width: { size: 9360, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: "FFF3D6" },
    margins: { top: 120, bottom: 120, left: 160, right: 160 },
    children: [
      new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: "▶ " + title, bold: true, size: 20 })] }),
      new Paragraph({ children: [new TextRun({ text: body, italics: true, size: 20 })] }),
    ],
  });
  return new Table({
    columnWidths: [9360],
    width: { size: 9360, type: WidthType.DXA },
    rows: [new TableRow({ children: [cell] })],
  });
}
function spacer() { return new Paragraph({ spacing: { after: 80 }, children: [] }); }

// generic table builder
function makeTable(headers, rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" };
  const borders = { top: border, bottom: border, left: border, right: border,
    insideHorizontal: border, insideVertical: border };
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((hh, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "1F3864" },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: hh, bold: true, color: "FFFFFF", size: 19 })] })],
    })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ri % 2 ? "F5F7FA" : "FFFFFF" },
      margins: { top: 50, bottom: 50, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: c, size: 19 })] })],
    })),
  }));
  return new Table({ columnWidths: widths, width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    borders, rows: [headerRow, ...bodyRows] });
}

const REPO = "https://github.com/manasa-manoj-nbr/fashion-retrieval";

// ---------- document body ---------------------------------------------------
const body = [];

// Title block
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Multimodal Fashion & Context Retrieval", bold: true, size: 40 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "Building a text-to-image search engine for fashion, tuned for compositional queries", italics: true, size: 22 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "Glance ML Internship Assignment", size: 20, color: "555555" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [
    new TextRun({ text: "Code: ", size: 20 }),
    new ExternalHyperlink({ link: REPO, children: [new TextRun({ text: REPO, style: "Hyperlink", size: 20 })] }),
  ] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: "Author: [ your name ]   ·   Date: [ date ]", size: 20, color: "555555" })] }));
body.push(new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1F3864" } }, spacing: { after: 160 }, children: [] }));

// 1. Overview
body.push(h1("1. The problem, and how I read it"));
body.push(p("The task is a natural-language image search over a fashion database: someone types “a person in a bright yellow raincoat” and the system should return the images that actually match. The naive version of this is easy – encode every image with CLIP, encode the query with CLIP, and rank by cosine similarity in a vector database. I built that first as a baseline, but the assignment is explicit that a plain CLIP demo is not the point, and I agree with the reasoning."));
body.push(p("The reason plain CLIP is not enough is worth stating precisely, because it drove every design choice I made. CLIP encodes a whole image into one vector. That vector is very good at telling you which concepts are present – red, blue, a shirt, trousers, an office – but it is weak at telling you how those concepts are bound together. “A red shirt with blue trousers” and “a blue shirt with red trousers” contain exactly the same words and the same colours, and a single-vector model scores them almost identically. Fashion search lives or dies on exactly this kind of distinction, and it is the failure the assignment calls out in its hint."));
body.push(p("So the goal I set myself was concrete: keep the zero-shot flexibility of CLIP, but fix the attribute-binding problem, and be honest about where the fix does and does not hold. The short version of my solution is decompose, bind, then verify. I break both the image and the query into garment-level parts, match part against part so a colour is tied to a specific garment, and then optionally use a small vision-question-answering model to double-check the top results."));

// 2. Approaches
body.push(h1("2. Approaches I considered, and the trade-offs"));
body.push(p("I looked at five broad options before settling on one. They are not mutually exclusive – my final system is really options 2, 3 and 4 layered together – but it is worth laying out what each one buys you and what it costs."));
body.push(makeTable(
  ["Approach", "What is good about it", "Where it falls down", "Good when"],
  [
    ["Vanilla CLIP + vector DB", "Trivial to build, fully zero-shot, strong general concepts", "Bag-of-concepts; cannot bind colour to garment; generic fashion vocabulary", "A quick baseline, or very broad queries"],
    ["FashionCLIP + vector DB", "Same simplicity, much better fashion vocabulary (fabric, cut, style)", "Still one vector per image, so binding is still weak", "Any fashion use case; a strictly better baseline"],
    ["Region decomposition + AND-scoring (chosen)", "Binds attributes to garments; fixes the colour-swap failure; still zero-shot", "Needs a segmenter; extra vectors per image; limited by segmenter granularity", "When compositional, multi-attribute queries matter"],
    ["VQA / VLM re-rank", "Grounded per-attribute checking; high precision on the shortlist", "Slow; only as good as the questions; cannot judge parts the segmenter cannot isolate", "Final-stage precision on a small candidate set"],
    ["Structured tagging + metadata filter", "Fast, interpretable, exact colour/category filters", "Brittle, needs a tagger, not zero-shot, fails on unseen descriptions", "Closed catalogues with clean labels"],
  ],
  [2100, 2620, 2620, 2020],
));
body.push(spacer());
body.push(p("The tagging route (option 5) is tempting because it is fast and exact, but it quietly reintroduces the thing the assignment warns against – it turns into keyword matching against a fixed label set and stops being zero-shot. I kept structured colour tags in my index, but only as a cheap secondary signal, never as the primary matcher."));

// 3. Chosen architecture
body.push(h1("3. My chosen architecture"));
body.push(p("The system splits cleanly into the two workflows the assignment asks for: an indexer (Part A) and a retriever (Part B). Logic is kept separate from data – every model name, fusion weight, path and garment mapping lives in a single config file, so the code has no magic numbers baked in."));

body.push(h2("Part A – the indexer"));
body.push(p("For every image I store three things rather than one:"));
body.push(bullet("A whole-image vector from FashionCLIP. This captures the scene and overall vibe – office, street, formal, casual – and drives fast recall.", "Global embedding. "));
body.push(bullet("I run a human-parsing segmentation model, crop each garment it finds (upper body, lower body, dress, and so on), and embed each crop separately with the same FashionCLIP. These are what make attribute binding possible.", "Per-garment region embeddings. "));
body.push(bullet("A cheap dominant-colour label per region, used as a robust secondary signal and for evaluation, not as the main matcher.", "Colour tag. "));
body.push(pr([
  new TextRun("All of this goes into ChromaDB, which I picked deliberately because the brief says to use the easiest convenient vector store rather than reinvent one. It is zero-config and uses an approximate-nearest-neighbour (HNSW) index under the hood, which is what lets the retrieval scale (see section 6). Nothing about the indexer reads dataset labels – it works on raw pixels only, which keeps the whole system zero-shot."),
]));

body.push(h2("Part B – the retriever"));
body.push(p("A query goes through four stages:"));
body.push(bullet("The natural-language string is parsed into a small structured object: a list of (colour, garment) pairs, plus any scene and style cues. “A red tie and a white shirt in a formal setting” becomes two pairs – red/tie and white/shirt – plus a formal style cue. This is rule-based so it runs offline and deterministically, with an optional LLM path for messier queries.", "Query decomposition. "));
body.push(bullet("The full query is matched against the global vectors to pull a candidate pool. This is the fast, high-recall step.", "Stage 1, recall. "));
body.push(bullet("For each (colour, garment) pair I score the pair against the matching garment region, then combine the pairs with a minimum – a logical AND. An image only scores well if every requested attribute is present on the right garment. This is the step a single global vector cannot do, and it is the core of the whole design.", "Stage 2, compositional binding. "));
body.push(bullet("A scene/style prompt is scored against the global vectors to capture the “where and vibe” part of the query. The three signals are then fused with configurable weights, with weight redistributed when a query has no colour/garment pairs so single-axis queries are not penalised.", "Stage 3, context + fusion. "));
body.push(pr([
  new TextRun({ text: "Optional re-rank. ", bold: true }),
  new TextRun("As a final step a local BLIP vision-question-answering model re-checks the shortlist by naming the colour of a garment (“what colour is the shirt?”) rather than being asked a leading yes/no question. It runs on the Colab GPU with no API key. Importantly, it skips any attribute the segmenter cannot isolate on its own – for example a tie and a shirt both fall in the same “upper” region, so rather than guess and corrupt the ranking, it leaves those queries on the fused order. This conservative behaviour is deliberate and is discussed again under shortcomings."),
]));

body.push(h2("How this handles the five evaluation queries"));
body.push(p("Walking through the required prompts is the clearest way to show how the parts fit:"));
body.push(bullet("attribute-specific, handled by the colour+garment pair on the upper region.", "“bright yellow raincoat” – "));
body.push(bullet("context-heavy, handled mostly by the global/scene signal (this is a weaker case for my data – see results).", "“professional business attire in an office” – "));
body.push(bullet("one binding pair (blue/shirt) plus a scene cue, fused together.", "“blue shirt on a park bench” – "));
body.push(bullet("pure style inference, handled by the global vector; this is one of the strongest results.", "“casual weekend outfit for a city walk” – "));
body.push(bullet("two binding pairs plus a formal cue – the hardest case, and an honest limitation because a tie is not separable by the segmenter.", "“red tie and white shirt in a formal setting” – "));

// 4. Results
body.push(h1("4. Results"));
body.push(p("I evaluated in two ways: one controlled quantitative test that isolates the binding claim, and a broad qualitative battery of queries. Both are reproducible from the notebook in the repo."));

body.push(h2("The headline: does binding actually improve?"));
body.push(p("The cleanest test I could design is a colour-swap test. I take images that have an upper garment of one colour and a lower garment of another, and for each I compare the correct description against the same description with the two colours swapped. A model that truly binds attributes ranks the correct one higher; a bag-of-concepts model is at chance. Crucially the two systems see the same images and the same colours – only the scoring method changes."));
body.push(makeTable(
  ["Scoring method", "Binding accuracy", "Notes"],
  [
    ["Global single-vector (vanilla-CLIP style)", "0.567", "Barely above the 0.500 chance line – confirms the failure the brief predicts"],
    ["Region AND-scoring (my pipeline)", "0.750", "Same 60 image samples; the only change is part-wise scoring"],
    ["Absolute improvement", "+0.183", "About a 32% relative gain from attribute binding alone"],
  ],
  [3400, 2200, 3760],
));
body.push(spacer());
body.push(placeholder(
  "PASTE FIGURE 1 HERE – quantitative results",
  "Screenshot the full output of  python -m eval.evaluate  (the colour-swap binding block AND the global-vs-full attribute table). This is your single most important piece of evidence, so give it room. Suggested caption: “Figure 1. Colour-swap binding test and per-query retrieval metrics. Region AND-scoring lifts binding accuracy from 0.567 to 0.750.”"
));

body.push(h2("Per-query retrieval metrics"));
body.push(p("On a small auto-labelled set I compared the global-only baseline against the full pipeline. The relevant sets are small, so treat these as supporting evidence rather than the headline; recall@5 is naturally low when many images are relevant, which is why precision@5 is the more meaningful column here."));
body.push(makeTable(
  ["Query", "R@5 global → full", "P@5 global → full", "MRR global → full"],
  [
    ["yellow raincoat", "0.50 → 0.50", "0.20 → 0.20", "0.20 → 1.00"],
    ["blue shirt", "0.00 → 0.33", "0.00 → 0.20", "0.07 → 0.25"],
    ["white shirt", "0.04 → 0.06", "0.40 → 0.60", "1.00 → 1.00"],
    ["red pants", "0.38 → 0.25", "0.60 → 0.40", "1.00 → 1.00"],
  ],
  [2400, 2320, 2320, 2320],
));
body.push(spacer());
body.push(p("The full pipeline wins clearly on three of the four queries – most strikingly it moves the correct yellow-raincoat image to rank 1 (MRR 0.20 to 1.00) and rescues the blue-shirt query from nothing. It loses on “red pants”, which I have left in rather than hide; it is a fusion-weighting artefact and with only four queries I did not want to tune the weights to them and overfit."));

body.push(h2("Qualitative results on the five evaluation queries"));
body.push(p("Screenshots of the top-5 image grids tell the real story. I have grouped them by how well they work, because being honest about the weak cases is part of the brief."));
body.push(placeholder(
  "PASTE FIGURE 2 HERE – the strong cases",
  "Paste the image grids for: “casual weekend outfit for a city walk” (this one is 5/5 – lead with it), plus two or three from the strengths gallery such as “a navy blue blazer”, “a leather biker jacket”, “an elegant evening gown” or “a floral summer dress”. Caption: “Figure 2. Style, colour and garment queries return clean, on-target results.”"
));
body.push(spacer());
body.push(placeholder(
  "PASTE FIGURE 3 HERE – the binding swap (nice to include)",
  "Paste the two grids  grid('a red top and blue pants')  and  grid('a blue top and red pants')  side by side. Point out in the caption that the rank-1 image changes between them even though the palette is identical – that is binding working. Caption: “Figure 3. Swapping the colours changes the top result, despite an identical set of words.”"
));
body.push(spacer());
body.push(placeholder(
  "PASTE FIGURE 4 HERE – the honest limitation",
  "Paste the grids for “Professional business attire inside a modern office” and “A red tie and a white shirt in a formal setting”. In the caption, note that business attire is found but no true office interior exists in the data, and that the tie query surfaces red garments generally because a tie is not a separable region. Caption: “Figure 4. Scene-dependent and fine-grained accessory queries are the weak spots – discussed in section 7.”"
));
body.push(spacer());
body.push(p("My honest reading of the qualitative results: colour, garment-type and style queries are strong and confident; the compositional binding works at rank-1 and is backed by the quantitative test; scene-dependent queries (office, park) and fine-grained accessories (a tie) are the weak spots, and both trace back to the same two causes covered next."));

// 5. Scalability
body.push(h1("5. Would this scale to a million images?"));
body.push(p("Yes, and the design already assumes it. Three things matter:"));
body.push(bullet("I never do a brute-force cosine over the whole dataset. Chroma uses an HNSW approximate-nearest-neighbour index, so recall is sub-linear in the number of images. This is the main reason I used a real vector store instead of a NumPy matrix.", "Approximate search. "));
body.push(bullet("The expensive parts – region AND-scoring and the VQA re-rank – only ever run on a small candidate pool that the cheap global search narrows down to (a couple of hundred). So cost is bounded by the pool size, not the corpus size. Going from a thousand images to a million barely changes the per-query work after Stage 1.", "Two-stage funnel. "));
body.push(bullet("Indexing is a batched GPU job that runs once and is embarrassingly parallel across images. At larger scale I would quantise the vectors (product quantisation or int8) to cut memory, shard the store, and move from local Chroma to a distributed engine like Qdrant or Milvus – the retrieval logic does not change.", "Index-time cost. "));

// 6. Zero-shot
body.push(h1("6. Zero-shot capability"));
body.push(p("This was a first-class concern, not an afterthought. The indexer and retriever never read Fashionpedia's labels – they operate on raw pixels and free text only. I used the dataset's annotations in exactly one place, to build evaluation ground truth, and nowhere in the live path. That means the system runs on arbitrary, unlabelled images and handles descriptions that never appeared as a training label."));
body.push(p("In practice this held up well. Queries like “a polka dot blouse”, “a monochrome black outfit” and “a floral summer dress” – none of which are label words – returned confident, on-target results. That is the FashionCLIP backbone doing what CLIP-style models are good at, and it is the main reason I built on top of CLIP rather than a closed classifier."));
body.push(placeholder(
  "OPTIONAL FIGURE HERE – zero-shot",
  "If you have room, paste the grid for “a polka dot blouse” or “a monochrome black outfit” to make the zero-shot point visual."
));

// 7. Shortcomings
body.push(h1("7. Shortcomings, and how I would address them"));
body.push(p("The brief specifically asks for this, and it is the part I care most about getting right."));
body.push(bullet("the human-parsing model I use lumps everything on the upper body into one region, so it cannot separate a tie, a shirt and a blazer. That is why the “red tie and white shirt” query is my weakest result – the parts simply are not separable, so the binding has nothing fine-grained to bind to. The fix is a finer-grained fashion segmenter or detector (for instance one trained on Fashionpedia's own garment categories, which do separate ties, collars and so on), which would slot straight into the same region-embedding interface.", "Coarse segmentation. "));
body.push(bullet("scene queries (office, park) underperform partly because my scene signal is just the global CLIP vector, and partly because Fashionpedia is mostly catalogue and runway imagery with few real office or park interiors. The data simply does not contain many of the scenes being asked for. I would add a dedicated place/scene classifier and, more importantly, index a dataset with genuine environmental variety.", "Weak scene understanding. "));
body.push(bullet("the fusion weights between composition, scene and global are sensible defaults I set by hand, not learned. With a labelled relevance set I would learn them, which would likely fix the one query (red pants) where the full pipeline currently loses to the baseline.", "Hand-set fusion weights. "));
body.push(bullet("VQA models love to answer “yes”. I already mitigate this by asking open-ended colour questions and by refusing to verify attributes the segmenter cannot isolate, but a stronger, more grounded VLM re-ranker would let me verify the fine-grained cases I currently skip.", "Re-ranker limits. "));

// 8. Future work
body.push(h1("8. Future work"));
body.push(h2("Adding locations, cities and weather"));
body.push(p("The architecture is already multi-signal, so adding new axes means adding new signals, not rebuilding. I would run a scene/place classifier and a weather tagger at index time and store their outputs both as filterable metadata (“city = Paris”, “weather = rainy”) and as extra embeddings that fold into the fusion step alongside the existing scene score. Hard facts like a city name are best as exact metadata filters; soft notions like “rainy-day outfit” are best as another soft-scored vector. Where images carry EXIF geotags those give location directly; otherwise a place-recognition model estimates it. The retriever's fusion stage is the natural home for weighting these against garment and colour signals."));
body.push(h2("Improving precision"));
body.push(bullet("the single highest-impact change, since it unlocks true garment-level binding for accessories.", "Finer segmentation – "));
body.push(bullet("fine-tune the FashionCLIP encoder with hard negatives (the colour-swapped pairs are ready-made hard negatives) so the embeddings themselves bind attributes better, not just the scoring on top.", "Hard-negative training – "));
body.push(bullet("replace the fixed fusion weights with learned ones and add a stronger cross-encoder or VLM re-ranker on the shortlist.", "Learned fusion and re-ranking – "));
body.push(bullet("expand queries into several phrasings and average, which smooths over prompt sensitivity.", "Query expansion – "));

// 9. Running it
body.push(h1("9. Running the code"));
body.push(p("The repository is organised as two clean modules – indexer/ and retriever/ – with a shared config, an evaluation harness, and a Colab notebook that reproduces every figure in this report. The full flow is four commands:"));
body.push(code("python -m eval.fetch_data --n 800        # download + subset Fashionpedia (236 MB)"));
body.push(code("python -m eval.smoke_test                # ~30s sanity check"));
body.push(code("python -m indexer.build --data data/images   # Part A: build the index"));
body.push(code("python -m retriever.query \"a red tie and a white shirt in a formal setting\" --k 5 --rerank"));
body.push(pr([
  new TextRun("Everything, including the indexing, retrieval, evaluation and the notebook, is at "),
  new ExternalHyperlink({ link: REPO, children: [new TextRun({ text: REPO, style: "Hyperlink" })] }),
  new TextRun("."),
]));

// ---------- assemble --------------------------------------------------------
const doc = new Document({
  creator: "Fashion Retrieval",
  title: "Multimodal Fashion & Context Retrieval",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1F3864" } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "2E5496" } },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 460, hanging: 260 } } } }],
    }],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1200, bottom: 1200, left: 1300, right: 1300 } } },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("docs/Fashion_Retrieval_Report.docx", buf);
  console.log("wrote docs/Fashion_Retrieval_Report.docx (" + buf.length + " bytes)");
});

/* A Tour of Machine Learning, chapter 04, Random forest: the live demos.
   Trees are grown in the browser as scikit-learn grows them (CART with the Gini impurity,
   thresholds halfway between neighbouring values, the largest weighted impurity decrease),
   and the forest as RandomForestClassifier does (a bootstrap sample per tree, sqrt(p) random
   features examined per split, the trees' class probabilities averaged). The rows are the
   notebook's: the 227 training patients first, then the 76 test patients. The forest's random
   draws are the browser's own, so its numbers vary a little from the notebook's.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { makeRng } = window.Stats;

/* ---------------- the data: 13 features, output, in the training set? ---------------- */
const FEATS = ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall"];
const ROWS = [[51,1,2,100,222,0,1,143,1,1.2,1,0,2,1,1],[66,0,0,178,228,1,1,165,1,1,1,2,3,0,1],[55,0,0,128,205,0,2,130,1,2,1,1,3,0,1],[60,1,0,140,293,0,0,170,0,1.2,1,2,3,0,1],[76,0,2,140,197,0,2,116,0,1.1,1,0,2,1,1],[66,1,1,160,246,0,1,120,1,0,1,3,1,0,1],[44,1,1,120,263,0,1,173,0,0,2,0,3,1,1],[50,0,0,110,254,0,0,159,0,0,2,0,2,1,1],[62,0,0,150,244,0,1,154,1,1.4,1,0,2,0,1],[41,1,1,120,157,0,1,182,0,0,2,0,2,1,1],[47,1,2,108,243,0,1,152,0,0,2,0,2,0,1],[66,0,3,150,226,0,1,114,0,2.6,0,0,2,1,1],[52,1,0,128,255,0,1,161,1,0,2,1,3,0,1],[49,1,2,120,188,0,1,139,0,2,1,3,3,0,1],[58,1,1,120,284,0,0,160,0,1.8,1,0,2,0,1],[56,1,0,132,184,0,0,105,1,2.1,1,1,1,0,1],[64,1,0,120,246,0,0,96,1,2.2,0,1,2,0,1],[58,1,0,128,259,0,0,130,1,3,1,2,3,0,1],[51,0,2,140,308,0,0,142,0,1.5,2,1,2,1,1],[56,0,0,134,409,0,0,150,1,1.9,1,2,3,0,1],[42,1,0,140,226,0,1,178,0,0,2,0,2,1,1],[38,1,2,138,175,0,1,173,0,0,2,4,2,1,1],[59,1,1,140,221,0,1,164,1,0,2,0,2,1,1],[70,1,1,156,245,0,0,143,0,0,2,0,2,1,1],[46,1,1,101,197,1,1,156,0,0,2,0,3,1,1],[64,1,2,140,335,0,1,158,0,0,2,0,2,0,1],[29,1,1,130,204,0,0,202,0,0,2,0,2,1,1],[38,1,3,120,231,0,1,182,1,3.8,1,0,3,0,1],[51,1,0,140,298,0,1,122,1,4.2,1,3,3,0,1],[64,0,2,140,313,0,1,133,0,0.2,2,0,3,1,1],[57,1,0,140,192,0,1,148,0,0.4,1,0,1,1,1],[56,1,3,120,193,0,0,162,0,1.9,1,0,3,1,1],[63,1,0,130,254,0,0,147,0,1.4,1,1,3,0,1],[50,0,1,120,244,0,1,162,0,1.1,2,0,2,1,1],[52,1,1,120,325,0,1,172,0,0.2,2,0,2,1,1],[41,1,1,135,203,0,1,132,0,0,1,0,1,1,1],[65,0,2,155,269,0,1,148,0,0.8,2,0,2,1,1],[65,1,0,110,248,0,0,158,0,0.6,2,2,1,0,1],[41,1,2,112,250,0,1,179,0,0,2,0,2,1,1],[46,1,0,120,249,0,0,144,0,0.8,2,0,3,0,1],[42,0,0,102,265,0,0,122,0,0.6,1,0,2,1,1],[74,0,1,120,269,0,0,121,1,0.2,2,1,2,1,1],[60,0,2,102,318,0,1,160,0,0,2,1,2,1,1],[66,0,2,146,278,0,0,152,0,0,1,1,2,1,1],[54,1,2,125,273,0,0,152,0,0.5,0,1,2,1,1],[63,1,0,140,187,0,0,144,1,4,2,2,3,0,1],[45,1,0,104,208,0,0,148,1,3,1,0,2,1,1],[59,1,0,138,271,0,0,182,0,0,2,0,2,1,1],[34,1,3,118,182,0,0,174,0,0,2,0,2,1,1],[62,0,2,130,263,0,1,97,0,1.2,1,1,3,0,1],[43,1,0,120,177,0,0,120,1,2.5,1,0,3,0,1],[62,1,0,120,267,0,1,99,1,1.8,1,2,3,0,1],[66,1,0,160,228,0,0,138,0,2.3,2,0,1,1,1],[77,1,0,125,304,0,0,162,1,0,2,3,2,0,1],[57,1,1,124,261,0,1,141,0,0.3,2,0,3,0,1],[48,1,2,124,255,1,1,175,0,0,2,2,2,1,1],[59,1,0,135,234,0,1,161,0,0.5,1,0,3,1,1],[51,0,2,130,256,0,0,149,0,0.5,2,0,2,1,1],[46,0,1,105,204,0,1,172,0,0,2,0,2,1,1],[42,1,2,120,240,1,1,194,0,0.8,0,0,3,1,1],[37,1,2,130,250,0,1,187,0,3.5,0,0,2,1,1],[58,1,2,132,224,0,0,173,0,3.2,2,2,3,0,1],[51,1,0,140,299,0,1,173,1,1.6,2,0,3,0,1],[44,1,0,110,197,0,0,177,0,0,2,1,2,0,1],[56,1,0,125,249,1,0,144,1,1.2,1,1,2,0,1],[48,1,1,110,229,0,1,168,0,1,0,0,3,0,1],[52,1,0,108,233,1,1,147,0,0.1,2,3,3,1,1],[40,1,0,110,167,0,0,114,1,2,1,0,3,0,1],[50,1,2,129,196,0,1,163,0,0,2,0,2,1,1],[44,0,2,118,242,0,1,149,0,0.3,1,1,2,1,1],[42,1,0,136,315,0,1,125,1,1.8,1,0,1,0,1],[62,0,0,124,209,0,1,163,0,0,2,0,2,1,1],[51,0,2,120,295,0,0,157,0,0.6,2,0,2,1,1],[57,0,0,128,303,0,0,159,0,0,2,1,2,1,1],[57,1,0,152,274,0,1,88,1,1.2,1,1,3,0,1],[54,1,2,150,232,0,0,165,0,1.6,2,0,3,1,1],[60,1,0,145,282,0,0,142,1,2.8,1,2,3,0,1],[52,1,2,138,223,0,1,169,0,0,2,4,2,1,1],[51,1,3,125,213,0,0,125,1,1.4,2,1,2,1,1],[66,1,0,112,212,0,0,132,1,0.1,2,1,2,0,1],[53,1,2,130,246,1,0,173,0,0,2,3,2,1,1],[64,1,0,145,212,0,0,132,0,2,1,2,1,0,1],[61,0,0,130,330,0,0,169,0,0,2,0,2,0,1],[57,1,0,110,335,0,1,143,1,3,1,1,3,0,1],[59,1,3,160,273,0,0,125,0,0,2,0,2,0,1],[57,1,2,128,229,0,0,150,0,0.4,1,1,3,0,1],[58,1,0,100,234,0,1,156,0,0.1,2,1,3,0,1],[58,1,0,146,218,0,1,105,0,2,1,1,3,0,1],[69,1,2,140,254,0,0,146,0,2,1,3,3,0,1],[50,0,2,120,219,0,1,158,0,1.6,1,0,2,1,1],[39,0,2,138,220,0,1,152,0,0,1,0,2,1,1],[44,1,1,120,220,0,1,170,0,0,2,0,2,1,1],[65,0,2,160,360,0,0,151,0,0.8,2,0,2,1,1],[54,0,1,132,288,1,0,159,1,0,2,1,2,1,1],[43,1,0,132,247,1,0,143,1,0.1,1,4,3,0,1],[41,0,2,112,268,0,0,172,1,0,2,0,2,1,1],[54,0,2,135,304,1,1,170,0,0,2,0,2,1,1],[60,0,0,150,258,0,0,157,0,2.6,1,2,3,0,1],[44,1,2,130,233,0,1,179,1,0.4,2,0,2,1,1],[55,0,0,180,327,0,2,117,1,3.4,1,0,2,0,1],[59,1,0,170,326,0,0,140,1,3.4,0,0,3,0,1],[64,0,0,130,303,0,1,122,0,2,1,2,2,1,1],[55,1,0,132,353,0,1,132,1,1.2,1,1,3,0,1],[43,0,0,132,341,1,0,136,1,3,1,0,3,0,1],[61,1,0,120,260,0,1,140,1,3.6,1,1,3,0,1],[45,0,1,130,234,0,0,175,0,0.6,1,0,2,1,1],[37,0,2,120,215,0,1,170,0,0,2,0,2,1,1],[59,1,0,110,239,0,0,142,1,1.2,1,1,3,0,1],[64,1,2,125,309,0,1,131,1,1.8,1,0,3,0,1],[46,0,0,138,243,0,0,152,1,0,1,0,2,1,1],[53,0,0,138,234,0,0,160,0,0,2,0,2,1,1],[58,1,2,105,240,0,0,154,1,0.6,1,0,3,1,1],[63,0,0,108,269,0,1,169,1,1.8,1,2,2,0,1],[44,1,2,140,235,0,0,180,0,0,2,0,2,1,1],[52,1,1,134,201,0,1,158,0,0.8,2,1,2,1,1],[58,0,1,136,319,1,0,152,0,0,2,2,2,0,1],[44,1,2,120,226,0,1,169,0,0,2,0,2,1,1],[64,1,3,110,211,0,0,144,1,1.8,1,0,2,1,1],[42,1,1,120,295,0,1,162,0,0,2,0,2,1,1],[46,1,2,150,231,0,1,147,0,3.6,1,0,2,0,1],[54,1,1,192,283,0,0,195,0,0,2,1,3,0,1],[41,1,0,110,172,0,0,158,0,0,2,0,3,0,1],[44,1,0,120,169,0,1,144,1,2.8,0,0,1,0,1],[68,1,2,118,277,0,1,151,0,1,2,1,3,1,1],[57,1,2,150,126,1,1,173,0,0.2,2,1,3,1,1],[44,1,1,130,219,0,0,188,0,0,2,0,2,1,1],[71,0,1,160,302,0,1,162,0,0.4,2,2,2,1,1],[65,1,0,120,177,0,1,140,0,0.4,2,0,3,1,1],[48,1,0,122,222,0,0,186,0,0,2,0,2,1,1],[35,1,0,120,198,0,1,130,1,1.6,1,0,3,0,1],[64,0,0,180,325,0,1,154,1,0,2,0,2,1,1],[38,1,2,138,175,0,1,173,0,0,2,4,2,1,1],[59,1,3,178,270,0,0,145,0,4.2,0,0,3,1,1],[70,1,0,130,322,0,0,109,0,2.4,1,3,2,0,1],[62,1,1,120,281,0,0,103,0,1.4,1,1,3,0,1],[61,1,0,138,166,0,0,125,1,3.6,1,1,2,0,1],[52,1,1,128,205,1,1,184,0,0,2,0,2,1,1],[43,1,2,130,315,0,1,162,0,1.9,2,1,2,1,1],[57,1,0,165,289,1,0,124,0,1,1,3,3,0,1],[55,1,0,140,217,0,1,111,1,5.6,0,0,3,0,1],[45,1,3,110,264,0,1,132,0,1.2,1,0,3,0,1],[57,0,0,140,241,0,1,123,1,0.2,1,0,3,0,1],[52,1,2,172,199,1,1,162,0,0.5,2,0,3,1,1],[52,1,0,125,212,0,1,168,0,1,2,2,3,0,1],[59,0,0,174,249,0,1,143,1,0,1,0,2,0,1],[54,1,0,124,266,0,0,109,1,2.2,1,1,3,0,1],[69,1,3,160,234,1,0,131,0,0.1,1,1,2,1,1],[57,1,0,130,131,0,1,115,1,1.2,1,1,3,0,1],[58,1,0,114,318,0,2,140,0,4.4,0,3,1,0,1],[70,1,0,145,174,0,1,125,1,2.6,0,0,3,0,1],[45,0,1,112,160,0,1,138,0,0,1,0,2,1,1],[58,1,1,125,220,0,1,144,0,0.4,1,4,3,1,1],[62,1,2,130,231,0,1,146,0,1.8,1,3,3,1,1],[64,1,3,170,227,0,0,155,0,0.6,1,0,3,1,1],[58,0,0,130,197,0,1,131,0,0.6,1,0,2,1,1],[41,0,1,126,306,0,1,163,0,0,2,0,2,1,1],[68,1,2,180,274,1,0,150,1,1.6,1,0,3,0,1],[60,1,0,117,230,1,1,160,1,1.4,2,2,3,0,1],[64,1,0,128,263,0,1,105,1,0.2,1,1,3,1,1],[68,0,2,120,211,0,0,115,0,1.5,1,0,2,1,1],[57,1,2,150,168,0,1,174,0,1.6,2,0,2,1,1],[69,0,3,140,239,0,1,151,0,1.8,2,2,2,1,1],[60,1,0,130,206,0,0,132,1,2.4,1,2,3,0,1],[39,1,0,118,219,0,1,140,0,1.2,1,0,3,0,1],[56,1,0,130,283,1,0,103,1,1.6,0,0,3,0,1],[50,1,0,150,243,0,0,128,0,2.6,1,0,3,0,1],[39,0,2,94,199,0,1,179,0,0,2,0,2,1,1],[65,1,0,135,254,0,0,127,0,2.8,1,1,3,0,1],[34,0,1,118,210,0,1,192,0,0.7,2,0,2,1,1],[58,1,2,140,211,1,0,165,0,0,2,0,2,1,1],[67,0,0,106,223,0,1,142,0,0.3,2,2,2,1,1],[63,0,2,135,252,0,0,172,0,0,2,0,2,1,1],[52,1,0,112,230,0,1,160,0,0,2,1,2,0,1],[45,1,0,115,260,0,0,185,0,0,2,0,2,1,1],[48,1,0,130,256,1,0,150,1,0,2,2,3,0,1],[60,1,0,125,258,0,0,141,1,2.8,1,1,3,0,1],[63,0,0,124,197,0,1,136,1,0,1,0,2,0,1],[47,1,2,138,257,0,0,156,0,0,2,0,2,1,1],[68,1,0,144,193,1,1,141,0,3.4,1,2,3,0,1],[55,1,0,160,289,0,0,145,1,0.8,1,1,3,0,1],[53,0,2,128,216,0,0,115,0,0,2,0,0,1,1],[59,1,0,140,177,0,1,162,1,0,2,1,3,0,1],[62,1,1,128,208,1,0,140,0,0,2,0,2,1,1],[54,1,0,140,239,0,1,160,0,1.2,2,0,2,1,1],[54,1,0,120,188,0,1,113,0,1.4,1,1,3,0,1],[62,0,0,160,164,0,0,145,0,6.2,0,3,3,0,1],[48,1,0,124,274,0,0,166,0,0.5,1,0,3,0,1],[58,1,0,128,216,0,0,131,1,2.2,1,3,3,0,1],[40,1,3,140,199,0,1,178,1,1.4,2,0,3,1,1],[47,1,0,112,204,0,1,143,0,0.1,2,0,2,1,1],[51,0,0,130,305,0,1,142,1,1.2,1,0,3,0,1],[51,1,0,140,261,0,0,186,1,0,2,0,2,1,1],[58,1,2,112,230,0,0,165,0,2.5,1,1,3,0,1],[53,1,0,123,282,0,1,95,1,2,1,2,3,0,1],[61,1,0,148,203,0,1,161,0,0,2,1,3,0,1],[41,0,1,130,204,0,0,172,0,1.4,2,0,2,1,1],[48,0,2,130,275,0,1,139,0,0.2,2,0,2,1,1],[65,0,0,150,225,0,0,114,0,1,1,3,3,0,1],[59,1,2,126,218,1,1,134,0,2.2,1,1,1,0,1],[55,1,1,130,262,0,1,155,0,0,2,0,2,1,1],[63,0,0,150,407,0,0,154,0,4,1,3,3,0,1],[47,1,2,130,253,0,1,179,0,0,2,0,2,1,1],[67,1,0,160,286,0,0,108,1,1.5,1,3,2,0,1],[56,0,1,140,294,0,0,153,0,1.3,1,0,2,1,1],[53,1,0,140,203,1,0,155,1,3.1,0,0,3,0,1],[56,1,1,120,236,0,1,178,0,0.8,2,0,2,1,1],[45,1,1,128,308,0,0,170,0,0,2,0,2,1,1],[43,1,0,150,247,0,1,171,0,1.5,2,0,2,1,1],[42,1,2,130,180,0,1,150,0,0,2,0,2,1,1],[58,0,2,120,340,0,1,172,0,0,2,0,2,1,1],[51,1,2,125,245,1,0,166,0,2.4,1,0,2,1,1],[51,1,2,94,227,0,1,154,1,0,2,1,3,1,1],[41,0,1,105,198,0,1,168,0,0,2,1,2,1,1],[41,1,2,130,214,0,0,168,0,2,1,0,2,1,1],[60,0,3,150,240,0,1,171,0,0.9,2,0,2,1,1],[55,0,1,132,342,0,1,166,0,1.2,2,0,2,1,1],[43,1,0,110,211,0,1,161,0,0,2,0,3,1,1],[49,0,0,130,269,0,1,163,0,0,2,0,2,1,1],[61,1,0,140,207,0,0,138,1,1.9,2,1,3,0,1],[47,1,0,110,275,0,0,118,1,1,1,1,2,0,1],[55,0,1,135,250,0,0,161,0,1.4,1,0,2,1,1],[58,0,0,170,225,1,0,146,1,2.8,1,2,1,0,1],[56,1,2,130,256,1,0,142,1,0.6,1,1,1,0,1],[71,0,2,110,265,1,0,130,0,0,2,1,2,1,1],[52,0,2,136,196,0,0,169,0,0.1,1,0,2,1,1],[44,0,2,108,141,0,1,175,0,0.6,1,0,2,1,1],[61,1,2,150,243,1,1,137,1,1,1,0,2,1,1],[70,1,2,160,269,0,1,112,1,2.9,1,1,3,0,0],[54,1,0,110,239,0,1,126,1,2.8,1,1,3,0,0],[57,1,1,154,232,0,0,164,0,0,2,1,2,0,0],[63,1,3,145,233,1,0,150,0,2.3,0,0,1,1,0],[56,1,1,130,221,0,0,163,0,0,2,0,3,1,0],[56,0,0,200,288,1,0,133,1,4,0,2,3,0,0],[53,0,0,130,264,0,0,143,0,0.4,1,0,2,1,0],[58,0,0,100,248,0,0,122,0,1,1,0,2,1,0],[45,1,0,142,309,0,0,147,1,0,1,3,3,0,0],[50,1,0,144,200,0,0,126,1,0.9,1,0,3,0,0],[35,1,0,126,282,0,0,156,1,0,2,0,3,0,0],[43,0,2,122,213,0,1,165,0,0.2,1,0,2,1,0],[49,0,1,134,271,0,1,162,0,0,1,0,2,1,0],[39,1,2,140,321,0,0,182,0,0,2,0,2,1,0],[57,1,0,150,276,0,0,112,1,0.6,1,1,1,0,0],[56,1,1,120,240,0,1,169,0,0,0,0,2,1,0],[60,0,2,120,178,1,1,96,0,0,2,0,2,1,0],[52,1,3,152,298,1,1,178,0,1.2,1,0,3,1,0],[46,0,2,142,177,0,0,160,1,1.4,0,0,2,1,0],[54,1,2,120,258,0,0,147,0,0.4,1,0,3,1,0],[51,1,2,110,175,0,1,123,0,0.6,2,0,2,1,0],[67,1,0,100,299,0,0,125,1,0.9,1,2,2,0,0],[57,0,0,120,354,0,1,163,1,0.6,2,0,2,1,0],[35,1,1,122,192,0,1,174,0,0,2,0,2,1,0],[45,0,0,138,236,0,0,152,1,0.2,1,0,2,1,0],[66,1,0,120,302,0,0,151,0,0.4,1,0,2,1,0],[61,0,0,145,307,0,0,146,1,1,1,0,3,0,0],[40,1,0,152,223,0,1,181,0,0,2,0,3,0,0],[63,0,1,140,195,0,1,179,0,0,2,2,2,1,0],[67,1,0,120,237,0,1,71,0,1,1,0,2,0,0],[60,1,2,140,185,0,0,155,0,3,1,0,2,0,0],[62,0,0,138,294,1,1,106,0,1.9,1,3,2,0,0],[57,1,0,132,207,0,1,168,1,0,2,0,3,1,0],[58,1,0,125,300,0,0,171,0,0,2,2,3,0,0],[58,1,0,150,270,0,0,111,1,0.8,2,0,3,0,0],[59,1,3,134,204,0,1,162,0,0.8,2,2,2,0,0],[42,1,3,148,244,0,0,178,0,0.8,2,2,2,1,0],[67,1,0,125,254,1,1,163,0,0.2,1,2,3,0,0],[54,1,1,108,309,0,1,156,0,0,2,0,3,1,0],[53,1,2,130,197,1,0,152,0,1.2,0,0,2,1,0],[49,1,1,130,266,0,1,171,0,0.6,2,0,2,1,0],[54,0,2,108,267,0,0,167,0,0,2,0,2,1,0],[63,1,0,130,330,1,0,132,1,1.8,2,3,3,0,0],[67,0,2,115,564,0,0,160,0,1.6,1,0,3,1,0],[50,1,2,140,233,0,1,163,0,0.6,1,1,3,0,0],[35,0,0,138,183,0,1,182,0,1.4,2,0,2,1,0],[60,1,0,130,253,0,1,144,1,1.4,2,1,3,0,0],[67,1,2,152,212,0,0,150,0,0.8,1,0,3,0,0],[44,1,0,112,290,0,0,153,0,0,2,1,2,0,0],[67,0,2,152,277,0,1,172,0,0,2,1,2,1,0],[49,1,2,118,149,0,0,126,0,0.8,2,3,2,0,0],[52,1,0,128,204,1,1,156,1,1,1,0,0,0,0],[42,0,2,120,209,0,1,173,0,0,1,0,2,1,0],[46,1,0,140,311,0,1,120,1,1.8,1,2,3,0,0],[53,1,0,142,226,0,0,111,1,0,2,0,3,1,0],[59,1,3,170,288,0,0,159,0,0.2,1,0,3,0,0],[57,1,0,110,201,0,1,126,1,1.5,1,0,1,1,0],[71,0,0,112,149,0,1,125,0,1.6,1,0,2,1,0],[54,1,0,122,286,0,0,116,1,3.2,1,2,2,0,0],[54,0,2,160,201,0,1,163,0,0,2,1,2,1,0],[61,1,3,134,234,0,1,145,0,2.6,1,2,2,0,0],[54,0,2,110,214,0,1,158,0,1.6,1,0,2,1,0],[48,1,1,130,245,0,0,180,0,0.2,1,0,2,1,0],[67,1,0,120,229,0,0,129,1,2.6,1,2,3,0,0],[59,1,2,150,212,1,1,157,0,1.6,2,0,2,1,0],[41,1,1,110,235,0,1,153,0,0,2,0,2,1,0],[52,1,3,118,186,0,0,190,0,0,1,0,1,1,0],[54,1,0,110,206,0,0,108,1,0,1,1,2,0,0],[59,1,0,164,176,1,0,90,0,1,1,2,1,0,0],[43,1,0,115,303,0,1,181,0,1.2,1,0,2,1,0],[62,0,0,140,268,0,0,160,0,3.6,0,2,2,0,0],[62,0,0,140,394,0,0,157,0,1.2,1,0,2,1,0],[57,0,1,130,236,0,0,174,0,0,1,1,2,0,0],[65,0,2,140,417,1,0,157,0,0.8,2,1,2,1,0],[58,0,3,150,283,1,0,162,0,1,2,0,2,1,0],[65,1,3,138,282,1,0,174,0,1.4,1,1,2,0,0]];
const P = FEATS.length;
const X = ROWS.map(r => r.slice(0, P)), Y = ROWS.map(r => r[P]);
const TR = ROWS.map((r, i) => i).filter(i => ROWS[i][P + 1]), TE = ROWS.map((r, i) => i).filter(i => !ROWS[i][P + 1]);

/* ---------------- CART, as scikit-learn grows it ---------------- */
const gini = (a, b) => { const n = a + b; return n ? 1 - (a / n) ** 2 - (b / n) ** 2 : 0; };
/* the best split of the rows idx (with repeats), examining features in the order given until
   `m` non-constant ones have been tried; returns null when no feature splits the node */
function bestSplit(idx, order, m) {
  const n = idx.length; let c1 = 0; for (const i of idx) c1 += Y[i]; const c0 = n - c1, G = gini(c0, c1);
  let best = null, tried = 0;
  for (const f of order) {
    const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
    if (X[s[0]][f] === X[s[n - 1]][f]) continue;
    tried++;
    let l0 = 0, l1 = 0;
    for (let k = 0; k < n - 1; k++) {
      const i = s[k]; if (Y[i]) l1++; else l0++;
      if (X[s[k + 1]][f] === X[i][f]) continue;
      const nl = k + 1, g = G - nl / n * gini(l0, l1) - (n - nl) / n * gini(c0 - l0, c1 - l1);
      if (!best || g > best.gain + 1e-12) best = { f, t: (X[i][f] + X[s[k + 1]][f]) / 2, gain: g };
    }
    if (tried >= m) break;
  }
  return best;
}
/* grow a tree on the rows idx; order(): the feature order to examine at each node; m: how many */
function grow(idx, maxDepth, order, m, depth = 0) {
  const n = idx.length; let c1 = 0; for (const i of idx) c1 += Y[i]; const c0 = n - c1;
  const node = { n, c: [c0, c1], G: gini(c0, c1) };
  if (c0 === 0 || c1 === 0 || n < 2 || (maxDepth != null && depth >= maxDepth)) return node;
  const sp = bestSplit(idx, order(), m);
  if (!sp) return node;
  node.f = sp.f; node.t = sp.t; node.gain = sp.gain;
  node.left = grow(idx.filter(i => X[i][sp.f] <= sp.t), maxDepth, order, m, depth + 1);
  node.right = grow(idx.filter(i => X[i][sp.f] > sp.t), maxDepth, order, m, depth + 1);
  return node;
}
function leafOf(node, x) { while (node.left) node = x[node.f] <= node.t ? node.left : node.right; return node; }
const proba = (tree, x) => { const l = leafOf(tree, x); return l.c[1] / l.n; };
/* mean decrease in impurity, normalised to sum to 1 for this tree */
function importance(tree) {
  const imp = new Float64Array(P), N = tree.n;
  (function walk(nd) { if (!nd.left) return; imp[nd.f] += nd.n / N * nd.gain; walk(nd.left); walk(nd.right); })(tree);
  const s = imp.reduce((a, b) => a + b, 0); return s > 0 ? imp.map(v => v / s) : imp;
}
const ALL = () => FEATS.map((_, j) => j);
function shuffled(rng) { const a = ALL(); for (let i = P - 1; i > 0; i--) { const j = Math.floor(rng.uniform() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
/* a forest of B trees: bootstrap sample each, m random features per split, probabilities averaged */
function forest(B, maxDepth, m, seed) {
  const rng = makeRng(seed), trees = [], bags = [];
  for (let b = 0; b < B; b++) {
    const bag = Array.from({ length: TR.length }, () => TR[Math.floor(rng.uniform() * TR.length)]);
    trees.push(grow(bag, maxDepth, () => shuffled(rng), m)); bags.push(new Set(bag));
  }
  return { trees, bags, m, maxDepth };
}
const acc = (idx, p) => idx.reduce((s, i, k) => s + ((p[k] > 0.5 ? 1 : 0) === Y[i] ? 1 : 0), 0) / idx.length;

/* =====================================================================
   THE DEMO: ONE TREE AGAINST A FOREST
   ===================================================================== */
const BS = [1, 2, 3, 5, 10, 25, 50, 100, 200, 300], hero = { bi: 9, depth: null, m: 3, seed: 42, cache: {} };
function heroForest() {
  const key = `${hero.depth}|${hero.m}|${hero.seed}`;
  if (hero.cache[key]) return hero.cache[key];
  const F = forest(300, hero.depth, hero.m, hero.seed), ptr = F.trees.map(t => TR.map(i => proba(t, X[i]))), pte = F.trees.map(t => TE.map(i => proba(t, X[i])));
  // accuracies of the first k trees, for every k, from running sums of the probabilities
  const sTr = new Float64Array(TR.length), sTe = new Float64Array(TE.length), oobS = new Float64Array(TR.length), oobN = new Float64Array(TR.length);
  const curve = [];
  for (let k = 0; k < 300; k++) {
    ptr[k].forEach((v, i) => { sTr[i] += v; if (!F.bags[k].has(TR[i])) { oobS[i] += v; oobN[i]++; } }); pte[k].forEach((v, i) => { sTe[i] += v; });
    let ok = 0, nn = 0; TR.forEach((i, q) => { if (oobN[q]) { nn++; if (((oobS[q] / oobN[q]) > 0.5 ? 1 : 0) === Y[i]) ok++; } });
    curve.push({ tr: acc(TR, Array.from(sTr, v => v / (k + 1))), te: acc(TE, Array.from(sTe, v => v / (k + 1))), oob: nn ? ok / nn : NaN, oobCover: nn });
  }
  const imps = F.trees.map(importance), imp = FEATS.map((_, j) => imps.reduce((s, v) => s + v[j], 0) / 300);
  return (hero.cache[key] = { F, curve, imp });
}
const single = { cache: {} };
function singleTree(depth) { const k = String(depth); return single.cache[k] || (single.cache[k] = grow(TR, depth, ALL, P)); }
function drawHero() {
  const H = heroForest(), B = BS[hero.bi], c = H.curve[B - 1], tree = singleTree(hero.depth);
  const tTr = acc(TR, TR.map(i => proba(tree, X[i]))), tTe = acc(TE, TE.map(i => proba(tree, X[i])));
  $("#hBV").textContent = String(B); $("#hDV").textContent = hero.depth == null ? "unlimited" : String(hero.depth); $("#hMV").textContent = `${hero.m} of 13`;
  const cv = $("#curveC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.35 : 2.3, { x: [0, Math.log10(300)], y: [0.6, 1.0], xlabel: "trees in the forest (log scale)", ylabel: "accuracy", xticks: [0, 1, 2, Math.log10(300)], xfmt: v => String(Math.round(10 ** v)), yticks: [0.6, 0.7, 0.8, 0.9, 1.0] });
  const { ctx, Xp, Yp } = { ctx: fr.ctx, Xp: fr.X, Yp: fr.Y }, ks = H.curve.map((_, k) => Math.log10(k + 1));
  ctx.strokeStyle = alpha("--muted", 0.8); ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]); seg(ctx, fr.pad.l, Yp(tTe), fr.pad.l + fr.iw, Yp(tTe)); ctx.setLineDash([]);
  ctx.fillStyle = css("--muted"); ctx.font = "600 10px " + css("--font-ui"); ctx.textAlign = "left"; ctx.fillText("one tree, test", fr.pad.l + 4, Yp(tTe) - 4);
  plotLine(fr, ks, H.curve.map(q => q.tr), { color: css("--post"), width: 1.6, dash: [2, 3] });
  plotLine(fr, ks, H.curve.map(q => q.oob), { color: css("--c4"), width: 1.8 });
  plotLine(fr, ks, H.curve.map(q => q.te), { color: css("--warm"), width: 2.4 });
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(Xp(Math.log10(B)), Yp(c.te), 5, 0, 2 * Math.PI); ctx.fill();
  $("#rTree").textContent = `${fmt(tTr, 3)} / ${fmt(tTe, 3)}`; $("#rFor").textContent = `${fmt(c.tr, 3)} / ${fmt(c.te, 3)}`;
  $("#rOob").textContent = isFinite(c.oob) ? `${fmt(c.oob, 3)} (${c.oobCover} patients)` : "–";
  const d = Math.round((c.te - tTe) * 76);
  let text = `<b>${B} tree${B > 1 ? "s" : ""}</b>, ${hero.depth == null ? "unlimited depth" : `depth at most ${hero.depth}`}, ${hero.m} candidate features per split: the forest gets <b>${Math.round(c.te * 76)} of 76</b> test patients right (${fmt(c.te, 3)}), the single tree ${Math.round(tTe * 76)} (${fmt(tTe, 3)}): ${d > 0 ? `${d} more` : d < 0 ? `${-d} fewer` : "the same"}. `;
  if (hero.depth == null || hero.depth >= 5) text += `(The notebook's unrestricted tree scores 0.724: its deepest splits have exact ties between equally good questions, which scikit-learn breaks in its own random order and this page in feature order; same training accuracy, same ${(function n(nd) { return nd.left ? n(nd.left) + n(nd.right) : 1; })(tree)} leaves, a few test patients apart. That is the variance of a single tree, seen from another angle.) `;
  if (B === 1) text += "One bootstrap tree alone is no better than the plain tree: it saw fewer distinct patients, and only 3 features per split. The gain comes from averaging. ";
  else if (B < 10) text += "A few trees already smooth out part of the noise; the curve is still jumpy. ";
  else text += `Each test patient is 1.3 points of accuracy, so the wiggles of the curve beyond about 25 trees are one or two patients, not a trend. `;
  text += `The out-of-bag estimate, from each patient's trees that never saw it, is ${isFinite(c.oob) ? fmt(c.oob, 3) : "not yet available"}: a free test set, close to the real one.`;
  if (hero.m === 13) text += " With all 13 features at every split the trees differ only through their bootstrap samples, so they are more alike and the average gains less.";
  caption($("#heroStatus"), text);
}
$("#hB").addEventListener("input", e => { hero.bi = +e.target.value; drawHero(); });
$("#hD").addEventListener("input", e => { hero.depth = +e.target.value >= 12 ? null : +e.target.value; drawHero(); });
$("#hM").addEventListener("input", e => { hero.m = +e.target.value; drawHero(); });
$("#hNew").addEventListener("click", () => { hero.seed++; drawHero(); drawImp(); });
$("#hReset").addEventListener("click", () => { hero.bi = 9; hero.depth = null; hero.m = 3; $("#hB").value = "9"; $("#hD").value = "12"; $("#hM").value = "3"; drawHero(); drawImp(); });

/* =====================================================================
   PART 1: THE GINI DECREASE OF A SPLIT (the hand example's node: 5 with, 5 without)
   ===================================================================== */
const ent = (a, b) => { const n = a + b; let h = 0; for (const c of [a, b]) if (c) h -= c / n * Math.log2(c / n); return h; };
function drawGini() {
  const nl = +$("#gN").value, dl = Math.min(+$("#gD").value, nl, 5); if (dl !== +$("#gD").value) $("#gD").value = String(dl);
  $("#gD").max = String(Math.min(nl, 5)); $("#gNV").textContent = String(nl); $("#gDV").textContent = String(dl);
  const nr = 10 - nl, dr = 5 - dl, GL = gini(nl - dl, dl), GR = gini(nr - dr, dr), dG = 0.5 - nl / 10 * GL - nr / 10 * GR;
  const HL = ent(nl - dl, dl), HR = ent(nr - dr, dr), dH = 1 - nl / 10 * HL - nr / 10 * HR;
  const cv = $("#giniC"), fr = frame(cv, narrowOf(cv) ? 1.6 : 3, { x: [0, 1], y: [0, 1], xticks: false, yticks: false });
  const { ctx, X: Xp, Y: Yp } = fr, box = (x0, x1, c0, c1, label) => {
    const w = Xp(x1) - Xp(x0), n = c0 + c1, h = fr.ih * 0.55, y = Yp(0.85);
    ctx.fillStyle = alpha("--warm", 0.35); ctx.fillRect(Xp(x0), y, w * c1 / n, h);
    ctx.fillStyle = alpha("--post", 0.35); ctx.fillRect(Xp(x0) + w * c1 / n, y, w * c0 / n, h);
    ctx.strokeStyle = css("--rule"); ctx.strokeRect(Xp(x0), y, w, h);
    ctx.fillStyle = css("--ink"); ctx.font = "600 11px " + css("--font-ui"); ctx.textAlign = "center"; ctx.fillText(label, Xp((x0 + x1) / 2), y + h + 14);
  };
  box(0.02, 0.02 + 0.96 * nl / 10, nl - dl, dl, `left: ${dl} / ${nl - dl}, G = ${fmt(GL, 3)}`); box(0.02 + 0.96 * nl / 10, 0.98, nr - dr, dr, `right: ${dr} / ${nr - dr}, G = ${fmt(GR, 3)}`);
  const name = nl === 4 && dl === 1 ? " This is split A of the table." : nl === 5 && dl === 4 ? " This is split B, the best split of these 10 patients: each side is 4 to 1." : nl === 5 && dl === 1 ? " This is split B with the sides swapped: the same decrease." : "";
  caption($("#giniCap"), `Sending ${nl} patients left, ${dl} of them with label 1 (orange): [[\\Delta G = 0.5 - \\tfrac{${nl}}{10} \\times ${fmt(GL, 3)} - \\tfrac{${nr}}{10} \\times ${fmt(GR, 3)} = ${fmt(dG, 3)}]]. Entropy agrees: a gain of ${fmt(dH, 3)} bits.${name}${dG < 1e-9 ? " The children have the same mix as the parent, so the split gains nothing." : ""}`);
}
["#gN", "#gD"].forEach(id => $(id).addEventListener("input", drawGini));

/* =====================================================================
   PART 2: A DEPTH-LIMITED TREE, DRAWN
   ===================================================================== */
function drawTree() {
  const depth = +$("#tD").value, tree = singleTree(depth), cv = $("#treeC"), narrow = narrowOf(cv);
  $("#tDV").textContent = String(depth);
  const leaves = []; (function count(nd) { if (!nd.left) leaves.push(nd); else { count(nd.left); count(nd.right); } })(tree);
  const fr = frame(cv, narrow ? (depth >= 3 ? 0.9 : 1.3) : (depth >= 3 ? 1.5 : 2.4), { x: [0, 1], y: [0, 1], xticks: false, yticks: false, pad: { l: 4, r: 4, t: 8, b: 4 } });
  const { ctx, X: Xp, Y: Yp } = fr, font = narrow ? 9 : (depth >= 3 ? 10 : 11), lineH = font * 1.25;
  // boxes: a split shows its feature, its threshold and its counts (3 lines); a leaf its class and counts (2 lines)
  const boxH = { split: lineH * 3 + 6, leaf: lineH * 2 + 6 };
  let li = 0;
  (function place(nd, d) { if (!nd.left) { nd.x = (li + 0.5) / leaves.length; li++; } else { place(nd.left, d + 1); place(nd.right, d + 1); nd.x = (nd.left.x + nd.right.x) / 2; } nd.d = d; })(tree, 0);
  const slot = fr.iw / Math.max(leaves.length, 2), boxW = Math.max(44, Math.min(110, slot - 4)), splitW = Math.min(110, Math.max(boxW, 2 * slot - 8));
  // when the leaves do not fit in one row, alternate them between two rows; the levels are spaced so that every box fits
  const stagger = boxW + 4 > slot ? boxH.leaf + 6 : 0; leaves.forEach((lf, k) => { lf.dy = stagger * (k % 2); });
  const top = fr.pad.t + boxH.split / 2 + 2, bottom = fr.pad.t + fr.ih - boxH.leaf / 2 - stagger - 2;
  const levelY = d => top + (depth ? d / depth : 0) * (bottom - top);
  const cx = (nd, w) => clamp(Xp(nd.x), fr.pad.l + w / 2, fr.pad.l + fr.iw - w / 2), cy = nd => levelY(nd.d) + (nd.dy || 0);
  (function draw(nd) {
    const w = nd.left ? splitW : boxW, h = nd.left ? boxH.split : boxH.leaf, x = cx(nd, w), y = cy(nd);
    if (nd.left) { ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1.2; [nd.left, nd.right].forEach(ch => seg(ctx, x, y + h / 2, cx(ch, ch.left ? splitW : boxW), cy(ch) - (ch.left ? boxH.split : boxH.leaf) / 2)); draw(nd.left); draw(nd.right); }
    const p1 = nd.c[1] / nd.n, fill = nd.left ? css("--surface") : alpha(p1 > 0.5 ? "--warm" : "--post", 0.15 + 0.45 * Math.abs(p1 - 0.5) * 2);
    ctx.fillStyle = fill; ctx.strokeStyle = nd.left ? css("--ink") : alpha(p1 > 0.5 ? "--warm" : "--post", 0.9); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.roundRect(x - w / 2, y - h / 2, w, h, 5); ctx.fill(); ctx.stroke();
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const lines = nd.left ? [[FEATS[nd.f], 600], [`≤ ${fmt(nd.t, nd.f === 9 ? 2 : 1).replace(/\.?0+$/, "")}`, 600], [`${nd.c[1]} / ${nd.c[0]}`, 400]] : [[p1 > 0.5 ? "→ 1" : "→ 0", 600], [`${nd.c[1]} / ${nd.c[0]}`, 400]];
    lines.forEach(([txt, wt], k) => { ctx.font = `${wt} ${font}px ${css("--font-mono")}`; ctx.fillStyle = wt === 600 ? css("--ink") : css("--ink-2"); ctx.fillText(txt, x, y - h / 2 + 3 + lineH * (k + 0.5)); });
    ctx.textBaseline = "alphabetic";
  })(tree);
  const trA = acc(TR, TR.map(i => proba(tree, X[i]))), teA = acc(TE, TE.map(i => proba(tree, X[i])));
  caption($("#treeCap"), `Depth ${depth}: ${leaves.length} leaves, each labelled with its patients' outputs, <b>1 / 0</b>, and the class it predicts. Training accuracy ${fmt(trA, 3)}, test accuracy ${fmt(teA, 3)}. ` +
    (depth === 1 ? "The root asks about chest-pain type alone: cp = 0 against the rest." : depth === 3 ? "This is the depth-3 tree of exercise 2. Its second-level questions differ on the two sides: the tree is already tailoring itself to subgroups." : "Each extra level asks a more specific question of fewer patients."));
}
$("#tD").addEventListener("input", drawTree);

/* =====================================================================
   PART 3: BOOTSTRAP COVERAGE, AND THE VARIANCE OF AN AVERAGE OF CORRELATED TREES
   ===================================================================== */
const boot = { seed: 1 };
function drawBoot() {
  const rng = makeRng(boot.seed), n = TR.length, counts = new Int32Array(n);
  for (let k = 0; k < n; k++) counts[Math.floor(rng.uniform() * n)]++;
  const distinct = counts.filter(c => c > 0).length, maxc = Math.max(...counts), hist = new Int32Array(maxc + 1); for (const c of counts) hist[c]++;
  const cv = $("#bootC"), fr = frame(cv, narrowOf(cv) ? 1.6 : 3, { x: [-0.5, maxc + 0.5], y: [0, Math.max(...hist) * 1.15], xlabel: "times drawn", xticks: Array.from({ length: maxc + 1 }, (_, i) => i), yticks: [0, 40, 80] });
  const { ctx, X: Xp, Y: Yp } = fr;
  hist.forEach((h, c) => { ctx.fillStyle = c === 0 ? css("--warm") : css("--post"); ctx.fillRect(Xp(c - 0.35), Yp(h), Xp(c + 0.35) - Xp(c - 0.35), Yp(0) - Yp(h)); ctx.fillStyle = css("--ink-2"); ctx.font = "600 10px " + css("--font-ui"); ctx.textAlign = "center"; ctx.fillText(String(h), Xp(c), Yp(h) - 4); });
  caption($("#bootCap"), `One bootstrap sample of the 227 training patients: <b>${distinct} distinct patients</b> (${pct(distinct / n, 1)}) are in the bag, ${n - distinct} are out of it (${pct((n - distinct) / n, 1)}, against the expected [[(1 - \\tfrac{1}{227})^{227} = 0.367]]); ${hist[1]} were drawn once, ${hist[2] || 0} twice${maxc >= 3 ? `, ${hist.slice(3).reduce((a, b) => a + b, 0)} three times or more` : ""}.`);
}
$("#bootAgain").addEventListener("click", () => { boot.seed++; drawBoot(); });
function drawVar() {
  const rho = +$("#vR").value; $("#vRV").textContent = fmt(rho, 2);
  const cv = $("#varC"), fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [0, Math.log10(300)], y: [0, 1], xlabel: "trees B (log scale)", ylabel: "Var of the average, σ² = 1", xticks: [0, 1, 2, Math.log10(300)], xfmt: v => String(Math.round(10 ** v)), yticks: [0, 0.25, 0.5, 0.75, 1] });
  const Bs = Array.from({ length: 300 }, (_, k) => k + 1), { ctx, X: Xp, Y: Yp } = fr;
  [[0.5, "--muted"], [0.2, "--muted"]].forEach(([r, col]) => plotLine(fr, Bs.map(Math.log10), Bs.map(B => r + (1 - r) / B), { color: alpha(col, 0.6), width: 1.2, dash: [4, 4] }));
  plotLine(fr, Bs.map(Math.log10), Bs.map(B => rho + (1 - rho) / B), { color: css("--warm"), width: 2.4 });
  ctx.strokeStyle = alpha("--warm", 0.7); ctx.setLineDash([2, 3]); seg(ctx, fr.pad.l, Yp(rho), fr.pad.l + fr.iw, Yp(rho)); ctx.setLineDash([]);
  ctx.fillStyle = css("--muted"); ctx.font = "600 10px " + css("--font-ui"); ctx.textAlign = "right"; ctx.fillText("ρ = 0.5", fr.pad.l + fr.iw - 4, Yp(0.5) - 4); ctx.fillText("ρ = 0.2", fr.pad.l + fr.iw - 4, Yp(0.2) - 4);
  caption($("#varCap"), `With correlation [[\\rho = ${fmt(rho, 2)}]] between any two trees: 10 trees bring the variance to ${fmt(rho + (1 - rho) / 10, 3)}, 100 trees to ${fmt(rho + (1 - rho) / 100, 3)}, and no number of trees goes below the floor [[\\rho\\sigma^2 = ${fmt(rho, 2)}]]. ` + (rho < 0.05 ? "Nearly independent trees: averaging alone would remove almost all the variance, but bootstrap trees of one data set are never this different." : rho > 0.7 ? "Trees this alike gain little from averaging: lowering ρ, by random feature subsets, matters more than adding trees." : "Random feature subsets exist to push ρ down, which lowers the floor itself."));
}
$("#vR").addEventListener("input", drawVar);

/* =====================================================================
   PART 7: FEATURE IMPORTANCE, MEAN DECREASE IN IMPURITY, AND BY PERMUTATION
   ===================================================================== */
const NB_IMP = { cp: 0.157507, oldpeak: 0.123321, thall: 0.120078, thalachh: 0.114468, caa: 0.079881, chol: 0.078990, age: 0.073159, trtbps: 0.068857, slp: 0.063579, exng: 0.062523, sex: 0.030288, restecg: 0.019888, fbs: 0.007461 };
const perm = { result: null, key: "" };
function permImportance(H) {
  const key = `${hero.depth}|${hero.m}|${hero.seed}`; if (perm.key === key) return perm.result;
  const rng = makeRng(99), trees = H.F.trees, base = acc(TE, TE.map(i => trees.reduce((s, t) => s + proba(t, X[i]), 0) / 300)), out = new Float64Array(P), R = 10;
  for (let f = 0; f < P; f++) {
    for (let r = 0; r < R; r++) {
      const col = TE.map(i => X[i][f]); for (let k = col.length - 1; k > 0; k--) { const j = Math.floor(rng.uniform() * (k + 1)); [col[k], col[j]] = [col[j], col[k]]; }
      const rows = TE.map((i, k) => { const x = X[i].slice(); x[f] = col[k]; return x; });
      out[f] += (base - acc(TE, rows.map(x => trees.reduce((s, t) => s + proba(t, x), 0) / 300))) / R;
    }
  }
  perm.key = key; return (perm.result = out);
}
function drawImp() {
  const H = heroForest(), showPerm = $("#impPerm").getAttribute("aria-pressed") === "true", pi = showPerm ? permImportance(H) : null;
  const order = FEATS.map((_, j) => j).sort((a, b) => H.imp[b] - H.imp[a]);
  const cv = $("#impC"), fr = frame(cv, narrowOf(cv) ? 1.1 : 1.7, { x: [showPerm ? -0.05 : 0, 0.2], y: [-0.5, P - 0.5], xlabel: showPerm ? "importance (permutation: accuracy drop)" : "importance (mean decrease in impurity)", yticks: false, xticks: showPerm ? [-0.05, 0, 0.05, 0.1, 0.15, 0.2] : [0, 0.05, 0.1, 0.15, 0.2], pad: { l: 68 } });
  const { ctx, X: Xp, Y: Yp } = fr, rowH = (Yp(0) - Yp(1));
  if (showPerm) { ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1; seg(ctx, Xp(0), fr.pad.t, Xp(0), fr.pad.t + fr.ih); }
  order.forEach((j, r) => {
    const yc = Yp(P - 1 - r), v = showPerm ? pi[j] : H.imp[j];
    ctx.fillStyle = showPerm ? (v >= 0 ? css("--c4") : alpha("--c4", 0.45)) : css("--post");
    ctx.fillRect(Math.min(Xp(0), Xp(v)), yc - rowH * 0.32, Math.abs(Xp(v) - Xp(0)), rowH * 0.64);
    if (!showPerm) { ctx.fillStyle = css("--warm"); ctx.fillRect(Xp(NB_IMP[FEATS[j]]) - 1, yc - rowH * 0.42, 2, rowH * 0.84); }
    ctx.fillStyle = css("--ink"); ctx.font = "600 11px " + css("--font-mono"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(FEATS[j], Xp(showPerm ? -0.05 : 0) - 6, yc);
    ctx.textAlign = "left"; ctx.fillStyle = css("--ink-2"); ctx.font = "10px " + css("--font-mono"); ctx.fillText(fmt(v, 3), Math.max(Xp(v), Xp(0)) + 4, yc); ctx.textBaseline = "alphabetic";
  });
  const top3 = order.slice(0, 3).map(j => FEATS[j]).join(", ");
  caption($("#impCap"), showPerm
    ? `Permutation importance on the 76 test patients: shuffle one feature's column, and see how much the forest's test accuracy drops (10 shuffles each). Top: ${order.slice().sort((a, b) => pi[b] - pi[a]).slice(0, 3).map(j => FEATS[j]).join(", ")}. Several features score near zero or below: shuffling them changes nothing, or even helps by chance. With 76 patients, one patient is 1.3 points, so only the largest bars mean much.`
    : `Mean decrease in impurity for this forest (blue), with the notebook's values as orange ticks. Top: ${top3}. The order of the leaders is stable from one forest to the next; the middle of the list is not. Continuous features with many thresholds (chol, age, trtbps) get credit here that the permutation test below does not confirm.`);
}
$("#impPerm").addEventListener("click", () => { const b = $("#impPerm"); b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") === "true" ? "false" : "true"); drawImp(); });

/* =====================================================================
   start-up
   ===================================================================== */
function drawAll() { drawHero(); drawGini(); drawTree(); drawBoot(); drawVar(); drawImp(); }
drawAll();
onRedraw(drawAll);
})();

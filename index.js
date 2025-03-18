const {initializeApp} = require('firebase/app');
const {getFireStore,collection,getDocs,doc, setDoc, updateDoc, getDoc} = require("firebase/firestore")
const {getDatabase,ref,get} = require("firebase/database")
const {GoogleGenerativeAi} = require("@google/generative-ai")
const {FaissStore} = require("langchain/vectorstores/faiss")
const  {GeminiProEmbeddings} = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter")
const { loadQAStuffChain,RetrievalQAChain} = require("/langchain/chains")
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai")
const fs = require("fs")
require("dotenv").config()

const firebaseConfig = {
    apiKey: process.env.apiKey,
    authDomain: process.env.authDomain,
    databaseURL: process.env.databaseURL,
    projectId: process.env.projectId,
    storageBucket: process.env.storageBucket,
    messagingSenderId: process.env.measurementId,
    appId: process.env.appId,
    measurementId: process.env.measurementId
  };

const geminiApiKey = process.env.GEMINI_API_KEY;
const genAI = new GoogleGenerativeAi(geminiApiKey);
const model = genAI.getGenerativeModel({model:"gemini-pro"});
const embeddings = new GeminiProEmbeddings({
    GoogleGenerativeAI: genAI
})

const app = initializeApp(firebaseConfig)
const db = getDatabase(app)


async function loadDataFromDatabase(path){
    const dbRef = ref(db,path)
    try {
        const snapshot = await get(dbRef);

        if(snapshot.exists()){
            const data = snapshot.val();
            const documents = object.entries(data).map(([id,value])=>({id,...value}));
            return documents;
        }else{
            console.log("no data found at the specified path")
        }
    } catch (error) {
        
    }
}


async function createEmbeddings(documents){
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize:1000, chunkOverlap:0
    });

    const texts = [];
    const metadatas = [];
    for(const doc of documents){
        const splits = await textSplitter.splitText(doc.text);
        texts.push(...splits);
        for(let i=0;i<splits.length;i++){
            metadatas.push({...doc.metadatas,doc_id:doc.id})
        }
    }
    console.log("Creating vectorStore");
    const vectorStore = await FaissStore.fromTexts(texts,metadatas, embeddings);
    console.log("vectorStore created");
    return vectorStore
}

async function storeFaissIndexToFireBase(index,collectionName,documentId){
    const faissIndexBuffer = await index.serialize();
    const docRef= doc(db,collectionName,documentId);
    await setDoc(docRef,{
        indexData: Array.from(new Uint8Array(faissIndexBuffer))
    });
    console.log("index stored at firestore")
}


async function loadFaissIndexFromFirebase(collectionName,documentId){
    const docRef = doc(db,collectionName,documentId)
    const docSnap = await getDoc(docRef);
    if(docSnap.exists()){
        const data = docSnap.data();
        const indexData = new Uint8Array(data.indexData);
        const buffer = Buffer.from(indexData);

        const dirPath = "./faiss_index";

        if(!fs.existsSync(dirPath)){
            fs.mkdirSync(dirPath,{recursive: true})
        }
        const indexPath = path.join(dirPath, 'faiss.index');
        const ivfPath = path.join(dirPath,"faiss.ivf");
        try {
            fs.writeFileSync(indexPath,buffer);
            fs.writeFileSync(ivfPath,'');
            console.log("faiss index file created")
        } catch (error) {
            console.error("failed to create faiss file",error)
            throw error;
        }
        const loadVectorStore = await FaissStore.load(dirPath,embeddings);
        console.log("faiss index loaded from firebase");
        return loadVectorStore;
    }else{
        console.log("No Faiss index found in Firebase");
        return null
    }
}

async function answerQuestion(vectorStore, question) {
    const model = new ChatGoogleGenerativeAI({
        apiKey: geminiApiKey,
        modelName: "gemini-pro",
        tempertaure: 0.7
    })

    const chain = RetrievalQAChain.fromLLM(model,vectorStore.asRetrieve());
    const res = await chain.call({
        query: question
    })
    return res.text
}

async function runMain(question){
    try {
        const documents = await loadDataFromDatabase("/")
        const vectorStore = await createEmbeddings(documents);

        await storeFaissIndexToFireBase(vectorStore.index,'faiss_indexes','my-index-doc')

        const loadedVectorStore = await loadFaissIndexFromFirebase('faiss_indexes','my_index_doc')

        if(loadedVectorStore){
            const answer = await answerQuestion(loadedVectorStore,question);
            console.log("Answer:",answer)
        }else{
            console.log("could not load the faiss index from firebase")
        }
    } catch (error) {
        console.log.error("Error:",error)
    }
}
runMain("hello there")
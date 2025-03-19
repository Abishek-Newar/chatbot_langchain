const {initializeApp} = require('firebase/app');
const {getFirestore,collection,getDocs,doc, setDoc, updateDoc, getDoc} = require("firebase/firestore")
const {getDatabase,ref,get} = require("firebase/database")
const {GoogleGenerativeAI} = require("@google/generative-ai")
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const  {GoogleGenerativeAIEmbeddings} = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter")
const {getStorage, ref:storageRef,uploadBytes,getDownloadURL} = require("firebase/storage")
const fs = require("fs")
const path = require("path")
require("dotenv").config()

const firebaseConfig = {
    apiKey: process.env.apiKey,
    authDomain: process.env.authDomain,
    databaseURL: process.env.databaseURL,
    projectId: process.env.projectId,
    storageBucket: process.env.storageBucket,
    messagingSenderId: process.env.messagingSenderId,
    appId: process.env.appId,
    measurementId: process.env.measurementId
};



const geminiApiKey = process.env.GOOGLE_API_KEY ;
console.log(geminiApiKey)
const genAI = new GoogleGenerativeAI(geminiApiKey);
const model = genAI.getGenerativeModel({model:"gemini-pro"});
const embeddings = new GoogleGenerativeAIEmbeddings({
    GoogleGenerativeAI: genAI
})

const app = initializeApp(firebaseConfig)
const firestore = getFirestore(app);
const storage = getStorage(app)
const db = getDatabase(app)


async function loadDataFromDatabase(path){
    const dbRef = ref(db, path);
  try {
    const snapshot = await get(dbRef);
    if (snapshot.exists()) {
      const items = snapshot.val();
      const documents = [];
      for(const data of items){
        const conversations = data.conversations || []; 
      const initial_query = data.initial_query || "";
      const product_title = data.product_title || "";
      const ticket_created_at = data.ticket_created_at || "";
      const ticket_id = data.ticket_id || "";


      


      documents.push({
        id: `ticket-${ticket_id}`,
        text: initial_query,
        metadata: {
          product_title: product_title,
          ticket_created_at: ticket_created_at,
          type: "initial_query"
        }
      });

      conversations.forEach((conversation, index) => {
        documents.push({
          id: `ticket-${ticket_id}-conversation-${index}`,
          text: conversation.conversation_content,
          metadata: {
            ticket_id: ticket_id,
            conversation_created_at: conversation.conversation_created_at,
            type: "conversation"
          }
        });
      });
      }
      console.log("documents:",documents)
      return documents;
    } else {
      console.log("No data available at the specified path.");
      return [];
    }
  } catch (error) {
    console.error("Error fetching data from Realtime Database:", error);
    return [];
  }
}


async function createEmbeddings(documents, batchSize = 500) {
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 0,
    });

    const vectorStores = [];

    for (let i = 0; i < documents.length; i += batchSize) {
        console.log(`Processing batch ${i / batchSize + 1}`);
        const batchDocs = documents.slice(i, i + batchSize);

        const docs = [];
        for (const doc of batchDocs) {
            const splits = await textSplitter.splitText(doc.text);
            for (const split of splits) {
                docs.push({
                    pageContent: split,
                    metadata: { ...doc.metadata, doc_id: doc.id }
                });
            }
        }

        const embeddings = new GoogleGenerativeAIEmbeddings({
            apiKey: geminiApiKey,
            modelName: "embedding-001"
        });

        const vectorStore = await FaissStore.fromDocuments(docs, embeddings);
        vectorStores.push(vectorStore);
    }

    return vectorStores;
}

async function storeFaissIndexToFireBase(vectorStore, folderName, fileName) {
    const tempDir = './faiss_temp';
    await vectorStore.save(tempDir);

    const indexBuffer = fs.readFileSync(path.join(tempDir, 'faiss.index'));
    const docstoreBuffer = fs.readFileSync(path.join(tempDir, 'docstore.json'));

    const indexRef = storageRef(storage, `${folderName}/${fileName}-faiss.index`);
    const docstoreRef = storageRef(storage, `${folderName}/${fileName}-docstore.json`);
    await uploadBytes(indexRef, indexBuffer);
    await uploadBytes(docstoreRef, docstoreBuffer);


    const indexURL = await getDownloadURL(indexRef);
    const docstoreURL = await getDownloadURL(docstoreRef);

    console.log("FAISS index stored in Firebase Storage");

    return { indexURL, docstoreURL }
}

async function runMain(){
    try {
        const documents = await loadDataFromDatabase("/")

        const vectorStores = await createEmbeddings(documents, 500);

        for (let i = 0; i < vectorStores.length; i++) {
            const { indexURL, docstoreURL } = await storeFaissIndexToFireBase(vectorStores[i], 'faiss_indexes', `my-index-doc-${i}`);

            const docRef = doc(firestore, 'faiss_indexes', `my-index-doc-${i}`);
            await setDoc(docRef, { indexURL, docstoreURL });

            console.log(`Stored FAISS index URLs for batch ${i}`);
        }

        console.log("All embeddings stored");
    } catch (error) {
        console.error("Error:", error);
    }
}

runMain()